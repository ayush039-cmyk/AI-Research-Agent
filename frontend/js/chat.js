let activeConversationId = null;

function renderHistory(conversations) {
  const list = document.getElementById('history-list');
  list.innerHTML = '';

  if (!conversations.length) {
    list.innerHTML = '<p style="color: var(--text-muted); font-size: 0.9rem;">No chats yet.</p>';
    return;
  }

  conversations.forEach((conversation) => {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = `history-item${conversation.id === activeConversationId ? ' active' : ''}`;
    item.innerHTML = `
      <h5>${escapeHtml(conversation.title)}</h5>
      <small>${formatDate(conversation.updated_at)} · ${conversation.message_count} messages</small>
    `;
    item.addEventListener('click', () => openConversation(conversation.id));
    list.appendChild(item);
  });
}

function escapeHtml(text) {
  return text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function renderMessages(messages) {
  const container = document.getElementById('messages');
  const emptyState = document.getElementById('empty-state');
  container.innerHTML = '';

  if (!messages.length) {
    container.appendChild(emptyState);
    emptyState.hidden = false;
    return;
  }

  messages.forEach((message) => {
    container.appendChild(createMessageElement(message));
  });

  container.scrollTop = container.scrollHeight;
}

function createMessageElement(message) {
  const el = document.createElement('article');
  el.className = `message ${message.role}`;
  el.innerHTML = `
    ${escapeHtml(message.content)}
    <span class="message-meta">${message.role === 'user' ? 'You' : 'Research Agent'} · ${formatDate(message.created_at)}</span>
  `;
  return el;
}

async function loadConversations() {
  const conversations = await api('/api/conversations');
  renderHistory(conversations);
  return conversations;
}

async function openConversation(conversationId) {
  activeConversationId = conversationId;
  const data = await api(`/api/conversations/${conversationId}`);

  document.getElementById('chat-title').textContent = data.title;
  document.getElementById('delete-chat-btn').hidden = false;
  renderMessages(data.messages);
  await loadConversations();
}

async function createConversation() {
  const conversation = await api('/api/conversations', {
    method: 'POST',
    body: JSON.stringify({ title: 'New research chat' }),
  });

  await openConversation(conversation.id);
}

async function deleteConversation() {
  if (!activeConversationId) return;
  if (!window.confirm('Delete this chat?')) return;

  await api(`/api/conversations/${activeConversationId}`, { method: 'DELETE' });
  activeConversationId = null;
  document.getElementById('chat-title').textContent = 'Research chat';
  document.getElementById('delete-chat-btn').hidden = true;
  renderMessages([]);
  await loadConversations();
}

function setLoading(isLoading) {
  document.getElementById('typing-indicator').classList.toggle('visible', isLoading);
  document.getElementById('send-btn').disabled = isLoading;
  document.getElementById('message-input').disabled = isLoading;
}

document.addEventListener('DOMContentLoaded', async () => {
  requireAuth();

  const user = getUser();
  if (user) {
    document.getElementById('user-chip').textContent = user.name;
  }

  document.getElementById('logout-btn').addEventListener('click', logout);
  document.getElementById('new-chat-btn').addEventListener('click', createConversation);
  document.getElementById('delete-chat-btn').addEventListener('click', deleteConversation);

  const conversations = await loadConversations();
  if (conversations.length) {
    await openConversation(conversations[0].id);
  } else {
    await createConversation();
  }

  const form = document.getElementById('chat-form');
  form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const input = document.getElementById('message-input');
    const message = input.value.trim();
    if (!message || !activeConversationId) return;

    const container = document.getElementById('messages');
    const emptyState = document.getElementById('empty-state');
    if (emptyState) emptyState.hidden = true;

    container.appendChild(createMessageElement({
      role: 'user',
      content: message,
      created_at: new Date().toISOString(),
    }));

    input.value = '';
    setLoading(true);
    container.scrollTop = container.scrollHeight;

    try {
      const data = await api(`/api/conversations/${activeConversationId}/chat`, {
        method: 'POST',
        body: JSON.stringify({ message }),
      });

      container.appendChild(createMessageElement(data.assistant_message));
      document.getElementById('chat-title').textContent = (
        data.user_message.content.length > 60
          ? `${data.user_message.content.slice(0, 60)}...`
          : data.user_message.content
      );
      await loadConversations();
    } catch (error) {
      container.appendChild(createMessageElement({
        role: 'assistant',
        content: `Error: ${error.message}`,
        created_at: new Date().toISOString(),
      }));
    } finally {
      setLoading(false);
      container.scrollTop = container.scrollHeight;
    }
  });
});
