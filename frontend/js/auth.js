document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('login-form');
  if (!form) return;

  if (getToken()) {
    window.location.href = 'chat.html';
    return;
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const submitBtn = form.querySelector('button[type="submit"]');
    submitBtn.disabled = true;

    try {
      const payload = {
        email: form.email.value.trim(),
        password: form.password.value,
      };

      const data = await api('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify(payload),
      });

      setAuth(data.token, data.user);
      window.location.href = 'chat.html';
    } catch (error) {
      showAlert('auth-alert', error.message);
    } finally {
      submitBtn.disabled = false;
    }
  });
});
