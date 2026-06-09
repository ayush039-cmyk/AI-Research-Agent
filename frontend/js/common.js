const TOKEN_KEY = 'research_agent_token';
const USER_KEY = 'research_agent_user';

function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function setAuth(token, user) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

function getUser() {
  const raw = localStorage.getItem(USER_KEY);
  return raw ? JSON.parse(raw) : null;
}

function requireAuth() {
  if (!getToken()) {
    window.location.href = '/login';
  }
}

async function api(path, options = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };

  const token = getToken();
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const response = await fetch(path, {
    ...options,
    headers,
  });

  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (response.status === 401) {
    clearAuth();
    if (!window.location.pathname.startsWith('/login') && !window.location.pathname.startsWith('/signup')) {
      window.location.href = '/login';
    }
  }

  if (!response.ok) {
    const message = data?.detail || 'Something went wrong';
    throw new Error(typeof message === 'string' ? message : JSON.stringify(message));
  }

  return data;
}

function showAlert(elementId, message, type = 'error') {
  const el = document.getElementById(elementId);
  if (!el) return;
  el.textContent = message;
  el.className = `alert alert-${type}`;
  el.hidden = false;
}

function formatDate(iso) {
  const date = new Date(iso);
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function logout() {
  clearAuth();
  window.location.href = '/';
}
