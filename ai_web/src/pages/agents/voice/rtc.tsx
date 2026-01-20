import React, { useEffect, useMemo, useRef, useState } from "react";
import { Alert, Button, Card, Col, Form, Input, Row, Space, Tag, Typography, message } from "antd";
import VERTC_SDK from "@volcengine/rtc";

import { TOKEN_KEY } from "../../../authProvider";

const { Title, Text } = Typography;

type RtcProxyAction = "StartVoiceChat" | "StopVoiceChat" | "GenerateRtcToken" | "UpdateVoiceChat";

type RtcProxyResponse<T = any> = {
  success: boolean;
  message: string;
  data?: T;
};

type GenerateRtcTokenData = {
  token: string;
  appId: string;
  roomId: string;
  userId: string;
};

type SubtitleMessage = {
  id: string;
  ts: number;
  userId: string;
  text: string;
  raw?: any;
};

function getAuthHeaders(): HeadersInit {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function proxyFetch<T = any>(action: RtcProxyAction, params: Record<string, any>): Promise<RtcProxyResponse<T>> {
  const res = await fetch("/api/rtc/proxyFetch", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...getAuthHeaders() },
    body: JSON.stringify({ action, params }),
  });
  const data = (await res.json()) as RtcProxyResponse<T>;
  if (!res.ok) throw new Error((data as any)?.message || (data as any)?.detail || `HTTP ${res.status}`);
  return data;
}

function tlv2String(tlvBuffer: ArrayBufferLike): { type: string; value: string } {
  const typeBuffer = new Uint8Array(tlvBuffer, 0, 4);
  const lengthBuffer = new Uint8Array(tlvBuffer, 4, 4);
  const valueBuffer = new Uint8Array(tlvBuffer, 8);

  let type = "";
  for (let i = 0; i < typeBuffer.length; i++) type += String.fromCharCode(typeBuffer[i]!);

  const length = (lengthBuffer[0]! << 24) | (lengthBuffer[1]! << 16) | (lengthBuffer[2]! << 8) | lengthBuffer[3]!;
  const value = new TextDecoder().decode(valueBuffer.subarray(0, length));
  return { type, value };
}

async function requestMicPermission(): Promise<boolean> {
  try {
    if (!navigator.mediaDevices?.getUserMedia) return false;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach((t) => t.stop());
    return true;
  } catch {
    return false;
  }
}

function getOrCreateStableId(key: string): string {
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const v = (globalThis.crypto as any)?.randomUUID ? (globalThis.crypto as any).randomUUID() : `${Date.now()}_${Math.random()}`;
  localStorage.setItem(key, v);
  return v;
}

export const VoiceAgentRtc = () => {
  const [form] = Form.useForm();

  const engineRef = useRef<any>(null);
  const [status, setStatus] = useState<"idle" | "connecting" | "connected">("idle");
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [logs, setLogs] = useState<SubtitleMessage[]>([]);
  const logsRef = useRef<SubtitleMessage[]>([]);

  const appendLog = (m: Omit<SubtitleMessage, "id">) => {
    const msg: SubtitleMessage = { id: `${m.ts}_${Math.random()}`, ...m };
    logsRef.current = [...logsRef.current, msg].slice(-200);
    setLogs(logsRef.current);
  };

  const statusTag = useMemo(() => {
    if (status === "connected") return <Tag color="green">Connected</Tag>;
    if (status === "connecting") return <Tag color="gold">Connecting</Tag>;
    return <Tag>Idle</Tag>;
  }, [status]);

  const ensureEngine = (appId: string) => {
    if (engineRef.current) return engineRef.current;
    const VERTC: any = VERTC_SDK as any;
    engineRef.current = VERTC.createEngine(appId);
    return engineRef.current;
  };

  const cleanupLocal = async () => {
    try {
      if (engineRef.current) {
        try {
          await engineRef.current.stopAudioCapture();
        } catch {
          // ignore
        }
        try {
          engineRef.current.leaveRoom();
        } catch {
          // ignore
        }
      }
    } finally {
      setAudioEnabled(false);
      setStatus("idle");
    }
  };

  const onConnect = async () => {
    if (status !== "idle") return;
    setStatus("connecting");

    const vals = form.getFieldsValue();
    const roomId = String(vals.roomId || "").trim();
    const userId = String(vals.userId || "").trim();
    const taskId = String(vals.taskId || "").trim();
    const botId = String(vals.botId || "").trim();
    const welcomeSpeech = String(vals.welcomeSpeech || "").trim();

    try {
      if (!roomId || !userId || !taskId) throw new Error("roomId, userId, taskId are required");

      const tokenRes = await proxyFetch<GenerateRtcTokenData>("GenerateRtcToken", { roomId, userId });
      if (!tokenRes.success || !tokenRes.data?.token) throw new Error(tokenRes.message || "GenerateRtcToken failed");

      const { token, appId } = tokenRes.data;

      const engine = ensureEngine(appId);

      // Register a few useful listeners (best-effort; SDK event names can vary by version).
      const VERTC: any = VERTC_SDK as any;
      const events = VERTC?.events || {};
      const add = (evt: any, fn: (...args: any[]) => void) => {
        try {
          if (evt) engine.on(evt, fn);
        } catch {
          // ignore
        }
      };

      add(events.onError, (e: any) => appendLog({ ts: Date.now(), userId: "system", text: `onError: ${JSON.stringify(e)}` }));
      add(events.onAutoplayFailed, (e: any) =>
        appendLog({ ts: Date.now(), userId: "system", text: `Autoplay blocked. Click in the page and try again. ${JSON.stringify(e)}` })
      );
      add(events.onUserJoined, (e: any) => appendLog({ ts: Date.now(), userId: "system", text: `User joined: ${JSON.stringify(e)}` }));
      add(events.onUserLeave, (e: any) => appendLog({ ts: Date.now(), userId: "system", text: `User left: ${JSON.stringify(e)}` }));
      add(events.onRoomBinaryMessageReceived, (event: any) => {
        try {
          const buf: ArrayBuffer = event?.message || event?.data;
          const from = String(event?.userId || "unknown");
          if (!buf) return;
          const { type, value } = tlv2String(buf);
          if (type !== "subv") return;
          const parsed = JSON.parse(value);
          const data = parsed?.data?.[0] ?? parsed;
          const text =
            String(data?.text || data?.content || data?.utterance || "").trim() || JSON.stringify(data);
          appendLog({ ts: Date.now(), userId: String(data?.userId || from), text, raw: data });
        } catch (err) {
          appendLog({ ts: Date.now(), userId: "system", text: `subtitle parse failed: ${String(err)}` });
        }
      });

      // Join room
      try {
        engine.enableAudioPropertiesReport?.({ interval: 1000 });
      } catch {
        // ignore
      }
      await engine.joinRoom(
        token,
        roomId,
        { userId },
        {
          isAutoPublish: true,
          isAutoSubscribeAudio: true,
          roomProfileType: 5, // chat
        }
      );

      const ok = await requestMicPermission();
      if (!ok) throw new Error("Microphone permission denied (requires HTTPS or localhost)");

      await engine.startAudioCapture();
      setAudioEnabled(true);

      const startRes = await proxyFetch("StartVoiceChat", {
        roomId,
        userId,
        taskId,
        botId: botId || undefined,
        welcomeSpeech: welcomeSpeech || undefined,
      });
      if (!startRes.success) throw new Error(startRes.message || "StartVoiceChat failed");

      setStatus("connected");
      message.success("Voice chat connected");
    } catch (e: any) {
      message.error(e?.message || "Connect failed");
      await cleanupLocal();
    }
  };

  const onDisconnect = async () => {
    const vals = form.getFieldsValue();
    const roomId = String(vals.roomId || "").trim();
    const taskId = String(vals.taskId || "").trim();

    try {
      if (roomId && taskId) {
        // Best-effort stop on the server.
        await proxyFetch("StopVoiceChat", { roomId, taskId });
      }
    } catch (e: any) {
      appendLog({ ts: Date.now(), userId: "system", text: `StopVoiceChat failed: ${e?.message || String(e)}` });
    } finally {
      await cleanupLocal();
    }
  };

  const onToggleMic = async () => {
    const engine = engineRef.current;
    if (!engine) return;
    try {
      if (audioEnabled) {
        await engine.stopAudioCapture();
        setAudioEnabled(false);
        message.info("Microphone muted");
      } else {
        const ok = await requestMicPermission();
        if (!ok) throw new Error("Microphone permission denied");
        await engine.startAudioCapture();
        setAudioEnabled(true);
        message.info("Microphone unmuted");
      }
    } catch (e: any) {
      message.error(e?.message || "Failed to toggle microphone");
    }
  };

  useEffect(() => {
    // defaults
    const stableUserId = getOrCreateStableId("fina_demo_voice_user_id");
    const stableRoomId = localStorage.getItem("fina_demo_voice_room_id") || "fina_demo_voice_room";
    form.setFieldsValue({
      roomId: stableRoomId,
      userId: stableUserId,
      taskId: "voice_agent",
      botId: "",
      welcomeSpeech: "",
    });

    return () => {
      void cleanupLocal();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: "100%" }} size={16}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            Voice Agent · Volcengine RTC
          </Title>
          <Text type="secondary">Connect to a Volcengine RTC room and start VoiceChat (ASR/TTS/LLM) via the agent.</Text>
        </div>

        <Alert
          type="info"
          showIcon
          message="Backend requirements"
          description={
            <div>
              <div>
                Agent env vars: <Text code>VOLCENGINE_APP_ID</Text>, <Text code>VOLCENGINE_APP_KEY</Text>,{" "}
                <Text code>VOLC_ACCESSKEY</Text>, <Text code>VOLC_SECRETKEY</Text> (plus optional Coze/TTS/ASR vars).
              </div>
              <div>Microphone permission requires HTTPS or localhost.</div>
            </div>
          }
        />

        <Row gutter={[16, 16]}>
          <Col xs={24} md={10}>
            <Card
              title={
                <Space>
                  <span>Session</span>
                  {statusTag}
                </Space>
              }
              extra={
                <Space>
                  <Button type="primary" onClick={onConnect} disabled={status !== "idle"}>
                    Connect
                  </Button>
                  <Button danger onClick={onDisconnect} disabled={status === "idle"}>
                    Disconnect
                  </Button>
                  <Button onClick={onToggleMic} disabled={status !== "connected"}>
                    {audioEnabled ? "Mute" : "Unmute"}
                  </Button>
                </Space>
              }
            >
              <Form form={form} layout="vertical">
                <Form.Item label="Room ID" name="roomId" rules={[{ required: true }]}>
                  <Input
                    placeholder="room id"
                    onChange={(e) => localStorage.setItem("fina_demo_voice_room_id", e.target.value)}
                    disabled={status !== "idle"}
                  />
                </Form.Item>
                <Form.Item label="User ID" name="userId" rules={[{ required: true }]}>
                  <Input placeholder="user id" disabled={status !== "idle"} />
                </Form.Item>
                <Form.Item label="Task ID" name="taskId" rules={[{ required: true }]}>
                  <Input placeholder="task id (used by StartVoiceChat)" disabled={status !== "idle"} />
                </Form.Item>
                <Form.Item label="Bot ID (optional)" name="botId">
                  <Input placeholder="Coze BotId (optional)" disabled={status !== "idle"} />
                </Form.Item>
                <Form.Item label="Welcome speech (optional)" name="welcomeSpeech">
                  <Input.TextArea rows={3} placeholder="Override default welcome speech" disabled={status !== "idle"} />
                </Form.Item>
              </Form>
            </Card>
          </Col>

          <Col xs={24} md={14}>
            <Card title="RTC Logs / Subtitles" extra={<Text type="secondary">{logs.length.toLocaleString("en-US")} events</Text>}>
              <div style={{ maxHeight: 520, overflow: "auto", fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas" }}>
                {logs.length === 0 ? (
                  <Text type="secondary">No events yet.</Text>
                ) : (
                  logs.map((l) => (
                    <div key={l.id} style={{ padding: "6px 0", borderBottom: "1px solid rgba(0,0,0,0.06)" }}>
                      <Text type="secondary">{new Date(l.ts).toLocaleTimeString()}</Text> <Text strong>{l.userId}:</Text>{" "}
                      <Text>{l.text}</Text>
                    </div>
                  ))
                )}
              </div>
            </Card>
          </Col>
        </Row>
      </Space>
    </div>
  );
};

