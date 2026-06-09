import os
from datetime import datetime

import paths  # noqa: F401
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from auth_utils import create_access_token, get_current_user, hash_password, verify_password
from database import Conversation, Message, User, get_db, init_db
from graph import run_research
from paths import FRONTEND_DIR

app = FastAPI(title='AI Research Agent')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount('/static/css', StaticFiles(directory=os.path.join(FRONTEND_DIR, 'css')), name='css')
app.mount('/static/js', StaticFiles(directory=os.path.join(FRONTEND_DIR, 'js')), name='js')


class SignupRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ConversationCreate(BaseModel):
    title: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@app.on_event('startup')
def on_startup() -> None:
    init_db()


def page(name: str) -> HTMLResponse:
    path = os.path.join(FRONTEND_DIR, f'{name}.html')
    return FileResponse(path)


@app.get('/', response_class=HTMLResponse)
def landing_page() -> FileResponse:
    return page('index')


@app.get('/login', response_class=HTMLResponse)
def login_page() -> FileResponse:
    return page('login')


@app.get('/signup', response_class=HTMLResponse)
def signup_page() -> FileResponse:
    return page('signup')


@app.get('/chat', response_class=HTMLResponse)
def chat_page() -> FileResponse:
    return page('chat')


@app.post('/api/auth/signup')
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email.lower()).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Email already registered')

    user = User(
        name=payload.name.strip(),
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        'token': create_access_token(user.id),
        'user': {'id': user.id, 'name': user.name, 'email': user.email},
    }


@app.post('/api/auth/login')
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid email or password')

    return {
        'token': create_access_token(user.id),
        'user': {'id': user.id, 'name': user.name, 'email': user.email},
    }


@app.get('/api/auth/me')
def me(current_user: User = Depends(get_current_user)):
    return {'id': current_user.id, 'name': current_user.name, 'email': current_user.email}


@app.get('/api/conversations')
def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )

    return [
        {
            'id': c.id,
            'title': c.title,
            'updated_at': c.updated_at.isoformat(),
            'message_count': len(c.messages),
        }
        for c in conversations
    ]


@app.post('/api/conversations')
def create_conversation(
    payload: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversation = Conversation(
        user_id=current_user.id,
        title=(payload.title or 'New research chat').strip()[:255],
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return {
        'id': conversation.id,
        'title': conversation.title,
        'updated_at': conversation.updated_at.isoformat(),
        'message_count': 0,
    }


@app.get('/api/conversations/{conversation_id}')
def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Conversation not found')

    return {
        'id': conversation.id,
        'title': conversation.title,
        'updated_at': conversation.updated_at.isoformat(),
        'messages': [
            {
                'id': m.id,
                'role': m.role,
                'content': m.content,
                'created_at': m.created_at.isoformat(),
            }
            for m in sorted(conversation.messages, key=lambda item: item.created_at)
        ],
    }


@app.delete('/api/conversations/{conversation_id}')
def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Conversation not found')

    db.delete(conversation)
    db.commit()
    return {'ok': True}


@app.post('/api/conversations/{conversation_id}/chat')
def chat(
    conversation_id: int,
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Conversation not found')

    user_message = Message(conversation_id=conversation.id, role='user', content=payload.message.strip())
    db.add(user_message)

    if conversation.title == 'New research chat':
        conversation.title = payload.message.strip()[:60] + ('...' if len(payload.message.strip()) > 60 else '')

    history_messages = sorted(conversation.messages, key=lambda item: item.created_at)
    chat_history = []
    for message in history_messages[-8:]:
        if message.role == 'user':
            chat_history.append(HumanMessage(content=message.content))
        elif message.role == 'assistant':
            chat_history.append(AIMessage(content=message.content))

    try:
        report = run_research(payload.message.strip(), chat_history=chat_history)
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Research agent failed: {exc}',
        ) from exc

    assistant_message = Message(conversation_id=conversation.id, role='assistant', content=report)
    db.add(assistant_message)
    conversation.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user_message)
    db.refresh(assistant_message)

    return {
        'user_message': {
            'id': user_message.id,
            'role': user_message.role,
            'content': user_message.content,
            'created_at': user_message.created_at.isoformat(),
        },
        'assistant_message': {
            'id': assistant_message.id,
            'role': assistant_message.role,
            'content': assistant_message.content,
            'created_at': assistant_message.created_at.isoformat(),
        },
    }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
