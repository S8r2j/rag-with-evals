# Events

Live status updates for documents being processed. Code: `app/events/` and the `/events` route in `app/api/routes/documents.py`.

## What the client sees

`GET /api/v1/documents/{id}/events` returns a [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) stream. The first event is the current state, then one event per step, and the stream closes when the document is `ready` or `failed`:

```
event: status
data: {"status": "pending", ...}

event: status
data: {"status": "processing", "stage": "extracting", ...}

event: status
data: {"status": "processing", "stage": "embedding", "chunks_done": 40, "chunks_total": 83, ...}

event: status
data: {"status": "ready", "chunks_total": 83, ...}
```

`chunks_done / chunks_total` is enough to draw a progress bar.

## Why server-sent events

| Option | Why or why not |
|---|---|
| Polling `GET /documents/{id}` | Simple, but slow to notice changes and wasteful when repeated every second |
| WebSockets | Two-way, which is not needed here; more to run and secure |
| **Server-sent events** | One-way, server to client, over plain HTTP: exactly this case |

## How it works

1. The worker publishes an event to Redis after each step is saved, so an event never announces something the database does not show yet.
2. The API subscribes to the document's Redis channel and forwards events to the client.
3. The API subscribes **before** reading the current state from the database, so an update that lands in between is not lost.
4. The database is the source of truth. Events are only notifications, so a missed one does no harm: a reconnecting client gets the current state first.

## Details that keep it reliable

- **Heartbeats.** A comment line every 15 seconds stops proxies from closing a quiet connection.
- **Time limit.** A stream closes after 30 minutes; the client reconnects and gets a fresh snapshot. `retry: 3000` tells browsers to reconnect after 3 seconds.
- **No held database connection.** The connection is released before streaming starts, so many open streams do not exhaust the database pool.
- **Updates never break processing.** If Redis is unreachable, the worker logs it and carries on.
- **No proxy buffering** (`X-Accel-Buffering: no`), so events arrive as they happen.

## Calling it from a browser

The endpoint needs the `Authorization` header, like the rest of the API. The browser's built-in `EventSource` cannot send headers, so the web client reads the stream with `fetch`:

```js
const response = await fetch(`/api/v1/documents/${id}/events`, {
  headers: { Authorization: `Bearer ${token}` },
});
const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
// read lines; each "data: {...}" line is one status update
```

Putting the token in the URL instead would let it leak into server and proxy logs.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `SSE_HEARTBEAT_SECONDS` | 15 | Gap between heartbeats |
| `SSE_MAX_STREAM_SECONDS` | 1800 | Longest a single stream stays open |
