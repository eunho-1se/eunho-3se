import uvicorn
import requests
import pymupdf  # PyMuPDF
import pymupdf4llm
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Body, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Ancient Language Researcher Backend")

# --- 설정 및 데이터 저장소 (In-Memory) ---
RAG_SERVER_URL = "http://localhost:8001"  # RAG 서버 주소 (가정)

# 사용자 데이터 저장소 (DB 대용)
users = []

# 사용자별 업로드된 텍스트 저장소 (username -> text)
# 고대 언어 문서는 사용자마다 다를 수 있으므로 분리 저장
user_contexts = {}


# --- Pydantic 모델 ---
class User(BaseModel):
    username: str
    password: str


class QueryRequest(BaseModel):
    query: str


# --- 유틸리티 함수 ---
def get_current_user(request: Request):
    """쿠키에서 username을 가져와 로그인 여부 확인"""
    username = request.cookies.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    # 실제 존재하는 유저인지 확인
    if not any(u.username == username for u in users):
        raise HTTPException(status_code=401, detail="유효하지 않은 사용자입니다.")

    return username


# --- 1. 회원가입 (Sign Up) ---
@app.post("/sign")
def sign(user: User):
    # 중복 아이디 체크
    if any(u.username == user.username for u in users):
        raise HTTPException(status_code=400, detail="이미 존재하는 아이디입니다.")

    users.append(user)
    return {"message": f"환영합니다, {user.username}님! 고대 언어 연구소 가입 완료."}


# --- 2. 회원탈퇴 (Cancel Membership) ---
@app.post("/Cancel_membership")
def cancel_membership(request: Request, response: Response):
    username = get_current_user(request)

    # 리스트에서 유저 삭제
    global users
    users = [u for u in users if u.username != username]

    # 컨텍스트 데이터 삭제
    if username in user_contexts:
        del user_contexts[username]

    # 쿠키 삭제
    response.delete_cookie("username")
    return {"message": "회원 탈퇴가 완료되었습니다. 데이터가 안전하게 파기되었습니다."}


# --- 3. 로그인 (Login) ---
@app.post("/login")
def login(user: User):
    # 아이디/비번 확인
    ok = any(u.username == user.username and u.password == user.password for u in users)

    if not ok:
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 잘못되었습니다.")

    content = {"message": f"안녕하세요, {user.username} 연구원님."}
    response = JSONResponse(content=content)

    # 쿠키 설정 (보안을 위해 httponly=True 권장하지만 테스트용이라 생략 가능)
    response.set_cookie(key="username", value=user.username)
    return response


# --- 4. 로그아웃 (Logout) ---
@app.post("/logout")
def logout(response: Response):
    response.delete_cookie("username")
    return {"message": "로그아웃 되었습니다."}


# --- 5. 파일 업로드 & 텍스트 추출 (Upload) ---
@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    # 1) 사용자 인증
    username = get_current_user(request)

    # 2) 파일 유효성 검사
    filename = (file.filename or "").lower()
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다")

    # 3) 파일 읽기
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="빈 파일입니다")

    # 4) PyMuPDF4LLM을 사용하여 Markdown 추출
    full_text = ""
    try:
        # stream=pdf_bytes로 직접 메모리에서 오픈
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        full_text = pymupdf4llm.to_markdown(doc)
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 파싱 실패: {str(e)}")

    # 5) 사용자별 컨텍스트에 저장 (전역 변수 대신 사용)
    user_contexts[username] = full_text

    # 6) (선택) 업로드 시점에 RAG 서버로 데이터 전송 (임베딩 미리 하기 위함)
    # try:
    #     upload_to_rag(full_text)
    # except:
    #     pass # RAG 서버가 꺼져있어도 일단 업로드는 성공 처리

    return {
        "ok": True,
        "user": username,
        "message": "고대 문서가 성공적으로 해독(Markdown 변환)되었습니다.",
        "text_length": len(full_text),
        "preview": full_text[:200] + "..."  # 앞부분 미리보기
    }


# --- 외부 RAG 서버 통신 함수 ---
def upload_to_rag(full_text, chunk_size: int = 1024):
    """텍스트를 RAG 서버로 보내 임베딩(Vector DB 저장) 요청"""
    try:
        response = requests.post(
            f"{RAG_SERVER_URL}/upload",
            json={"full_text": full_text, "chunk_size": chunk_size},
            timeout=10
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"RAG Server Error: {e}")
        return {"error": "RAG 서버 연결 실패"}


def llm_response(query: str, context: str):
    """질문과 컨텍스트를 RAG/LLM 서버로 전송"""
    payload = {
        "query": query,
        "context": context  # 현재 사용자가 업로드한 문서 내용을 같이 보냄
    }
    try:
        response = requests.post(
            f"{RAG_SERVER_URL}/answer",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"answer": f"죄송합니다. 인공지능 모델 서버에 연결할 수 없습니다. ({str(e)})"}


# --- 6. 질문하기 (Query) ---
@app.post("/query")
def query(request: Request, q_req: QueryRequest):
    # 1) 사용자 인증
    username = get_current_user(request)

    # 2) 사용자가 업로드한 문맥 가져오기
    context = user_contexts.get(username)
    if not context:
        raise HTTPException(status_code=400, detail="먼저 고대 문서를 업로드해주세요.")

    user_query = q_req.query

    # 3) (옵션) 질문할 때마다 RAG에 최신 컨텍스트를 업데이트하거나,
    #    이미 Upload 단계에서 했다면 생략 가능. 여기서는 직접 컨텍스트를 보낸다고 가정.

    # 4) LLM 응답 요청
    lr = llm_response(user_query, context)

    return {
        "ok": True,
        "user": username,
        "query": user_query,
        "result": lr
    }


if __name__ == "__main__":
    # 실행: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)