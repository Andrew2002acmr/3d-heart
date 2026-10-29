// Install before the first navigation. Application requests cannot leave loopback.
// Synthetic E2E uses this same policy, independent of HTTP response headers.
export async function installLocalNetworkGuard(context, origin, onBlocked = () => {}) {
  const base = new URL(origin);
  if (base.hostname !== '127.0.0.1' || base.protocol !== 'http:' || base.origin !== origin) {
    throw new Error('An exact http://127.0.0.1:port origin is required');
  }
  await context.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.origin === origin) return route.continue();
    onBlocked(url.hostname);
    return route.abort('blockedbyclient');
  });
  await context.routeWebSocket('**/*', socket => {
    const url = new URL(socket.url());
    if (url.origin === origin.replace('http:', 'ws:')) socket.connectToServer();
    else { onBlocked(url.hostname); socket.close(); }
  });
}
