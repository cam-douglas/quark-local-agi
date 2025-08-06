#!/usr/bin/env python3
from fastapi import FastAPI, WebSocket
import uvicorn
from orchestrator import Orchestrator

app = FastAPI()
orch = Orchestrator()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        text = await ws.receive_text()
        result = orch.handle(text)

        # stream if generator
        if hasattr(result["result"], "__iter__") and not isinstance(result["result"], str):
            async for chunk in result["result"]:
                await ws.send_text(chunk)
            await ws.send_text("<END>")
        else:
            await ws.send_json(result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

