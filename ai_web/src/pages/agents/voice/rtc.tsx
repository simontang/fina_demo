import { Bubble } from "@ant-design/x";
import VERTC_SDK from "@volcengine/rtc";
import { Alert, Button, Form, Input, Modal, Space, Tag, Typography, message } from "antd";
import { useEffect, useMemo, useRef, useState } from "react";

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
  sentenceId?: number;
  sequence?: number;
  definite?: boolean;
  paragraph?: boolean;
  timestamp?: number;
};

type SessionInfo = {
  roomId: string;
  userId: string;
  taskId: string;
  voiceChatStarted: boolean;
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

function newId(): string {
  const c = globalThis.crypto as any;
  if (c?.randomUUID) return c.randomUUID();
  return `${Date.now()}_${Math.random()}`;
}

export const VoiceAgentRtc = () => {
  const [form] = Form.useForm();

  const engineRef = useRef<any>(null);
  const sessionRef = useRef<SessionInfo>({ roomId: "", userId: "", taskId: "", voiceChatStarted: false });
  const [status, setStatus] = useState<"idle" | "connecting" | "connected">("idle");
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [autoplayBlockedUserId, setAutoplayBlockedUserId] = useState<string>("");
  const [logs, setLogs] = useState<SubtitleMessage[]>([]);
  const logsRef = useRef<SubtitleMessage[]>([]);
  const logsContainerRef = useRef<HTMLDivElement>(null);
  const [isConnectModalVisible, setIsConnectModalVisible] = useState(false);
  const [botIdInput, setBotIdInput] = useState("");

  const messageListRef = useRef<SubtitleMessage[]>([]);

  const appendLog = (m: Omit<SubtitleMessage, "id">) => {
    const msg: SubtitleMessage = { id: `${m.ts}_${Math.random()}`, ...m };
    logsRef.current = [...logsRef.current, msg].slice(-200);
    setLogs(logsRef.current);
  };

  const updateSubtitleMessage = (payload: {
    userId: string;
    text: string;
    sentenceId?: number;
    sequence?: number;
    definite?: boolean;
    paragraph?: boolean;
  }) => {
    if (
      !payload.userId ||
      payload.text === undefined ||
      payload.sentenceId === undefined
    ) {
      return;
    }

    const now = Date.now();
    const currentList = [...messageListRef.current];

    const newMessage: SubtitleMessage = {
      id: `${now}_${Math.random()}`,
      ts: now,
      userId: payload.userId,
      text: payload.text,
      sentenceId: payload.sentenceId,
      sequence: payload.sequence,
      definite: payload.definite ?? false,
      paragraph: payload.paragraph ?? false,
      timestamp: now,
    };

    // Determine if the last message is completed (either definite or paragraph)
    const lastMsg = currentList.length
      ? currentList[currentList.length - 1]
      : null;
    const lastMsgCompleted = !!(
      lastMsg &&
      (lastMsg.definite || lastMsg.paragraph)
    );

    if (currentList.length) {
      // If the last message is completed OR the last message is from a different user,
      // push a new message. Otherwise update the last message in-place (continuation).
      if (lastMsgCompleted || !lastMsg || lastMsg.userId !== payload.userId) {
        currentList.push(newMessage);
      } else {
        // Update last message in-place. Keep sequence comparison if provided to avoid
        // regressions from out-of-order packets.
        const existingMsg = lastMsg;
        if (
          payload.sequence !== undefined &&
          existingMsg.sequence !== undefined
        ) {
          if (payload.sequence >= existingMsg.sequence) {
            existingMsg.text = payload.text;
            existingMsg.sequence = payload.sequence;
            existingMsg.paragraph = payload.paragraph ?? false;
            existingMsg.definite = payload.definite ?? false;
            existingMsg.sentenceId = payload.sentenceId;
            existingMsg.timestamp = now;
          }
        } else {
          existingMsg.text = payload.text;
          existingMsg.sequence = payload.sequence;
          existingMsg.paragraph = payload.paragraph ?? false;
          existingMsg.definite = payload.definite ?? false;
          existingMsg.sentenceId = payload.sentenceId;
          existingMsg.timestamp = now;
        }

        currentList[currentList.length - 1] = existingMsg;
      }
    } else {
      // First message — always push
      currentList.push(newMessage);
    }

    messageListRef.current = currentList;

    // Also update logs so subtitle messages appear in UI
    if (currentList.length) {
      const latestMessage = currentList[currentList.length - 1];

      // Check if we need to add a new log entry or update existing one
      const existingLogIndex = logsRef.current.findIndex(
        log => log.sentenceId === latestMessage.sentenceId && log.userId === latestMessage.userId
      );

      if (existingLogIndex >= 0) {
        // Update existing log entry
        const updatedLogs = [...logsRef.current];
        updatedLogs[existingLogIndex] = { ...latestMessage, id: updatedLogs[existingLogIndex].id };
        logsRef.current = updatedLogs;
        setLogs(updatedLogs);
      } else {
        // Add new log entry
        appendLog({
          ts: latestMessage.ts,
          userId: latestMessage.userId,
          text: latestMessage.text,
          raw: latestMessage.raw,
          sentenceId: latestMessage.sentenceId,
          sequence: latestMessage.sequence,
          definite: latestMessage.definite,
          paragraph: latestMessage.paragraph,
          timestamp: latestMessage.timestamp,
        });
      }
    }
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

  const resumePlay = (uid?: string) => {
    const engine = engineRef.current;
    if (!engine) return;
    try {
      if (uid) engine.play(uid);
      else engine.play();
      setAutoplayBlockedUserId("");
      appendLog({ ts: Date.now(), userId: "system", text: `Playback resumed${uid ? ` for ${uid}` : ""}` });
    } catch (e: any) {
      appendLog({ ts: Date.now(), userId: "system", text: `Resume playback failed: ${e?.message || String(e)}` });
    }
  };

  const stopRemoteVoiceChatBestEffort = async () => {
    const s = sessionRef.current;
    if (!s.voiceChatStarted) return;
    if (!s.roomId || !s.taskId) return;
    try {
      await proxyFetch("StopVoiceChat", { roomId: s.roomId, taskId: s.taskId });
    } catch (e: any) {
      appendLog({ ts: Date.now(), userId: "system", text: `StopVoiceChat failed: ${e?.message || String(e)}` });
    } finally {
      sessionRef.current = { ...s, voiceChatStarted: false };
    }
  };

  const cleanupLocal = async (opts?: { stopRemote?: boolean }) => {
    try {
      if (opts?.stopRemote) {
        await stopRemoteVoiceChatBestEffort();
      }
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
        try {
          const VERTC: any = VERTC_SDK as any;
          VERTC.destroyEngine?.(engineRef.current);
        } catch {
          // ignore
        } finally {
          engineRef.current = null;
        }
      }
    } finally {
      setAudioEnabled(false);
      setAutoplayBlockedUserId("");
      setStatus("idle");
    }
  };

  const onConnect = async (botId?: string) => {
    if (status !== "idle") return;
    // Clear logs on reconnect
    setLogs([]);
    logsRef.current = [];
    messageListRef.current = [];
    setStatus("connecting");

    const stableUserId = getOrCreateStableId("fina_demo_voice_user_id");
    const stableRoomId = localStorage.getItem("fina_demo_voice_room_id") || newId();
    const roomId = stableRoomId;
    const userId = stableUserId;
    const taskId = "voice_agent";

    try {
      sessionRef.current = { roomId, userId, taskId, voiceChatStarted: false };

      const tokenRes = await proxyFetch<GenerateRtcTokenData>("GenerateRtcToken", {
        roomId,
        userId,
      });
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
      add(events.onAutoplayFailed, (e: any) => {
        const uid = String(e?.userId || "");
        setAutoplayBlockedUserId(uid);
        appendLog({
          ts: Date.now(),
          userId: "system",
          text: `Autoplay blocked by browser. Click "Resume Playback" to continue. ${JSON.stringify(e)}`,
        });
      });
      add(events.onUserJoined, (e: any) => appendLog({ ts: Date.now(), userId: "system", text: `User joined: ${JSON.stringify(e)}` }));
      add(events.onUserLeave, (e: any) => appendLog({ ts: Date.now(), userId: "system", text: `User left: ${JSON.stringify(e)}` }));
      add(events.onUserPublishStream, (e: any) => {
        const uid = String(e?.userId || "");
        appendLog({ ts: Date.now(), userId: "system", text: `User published: ${JSON.stringify(e)}` });
        if (!uid) return;
        try {
          // Best-effort: ensure the bot welcome audio can be heard even if autoplay is flaky.
          engine.play(uid);
        } catch {
          setAutoplayBlockedUserId(uid);
        }
      });
      add(events.onRoomBinaryMessageReceived, (event: any) => {
        try {
          const buf: ArrayBuffer = event?.message || event?.data;
          const from = String(event?.userId || "unknown");
          if (!buf) return;
          const { type, value } = tlv2String(buf);
          if (type !== "subv") return;
          const parsed = JSON.parse(value);
          const data = parsed?.data?.[0] ?? parsed;
          const text = data.text ?? data.msg ?? "";
          const sentenceId = data.roundId;
          const sequence = data.sequence;
          if (!text || !from || sentenceId === undefined) {
            return;
          }
          updateSubtitleMessage({
            userId: String(data?.userId || from),
            text,
            sentenceId,
            sequence,
            definite: data.definite,
            paragraph: data.paragraph,
          });
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
      });
      if (!startRes.success) throw new Error(startRes.message || "StartVoiceChat failed");
      sessionRef.current = { roomId, userId, taskId, voiceChatStarted: true };

      setStatus("connected");
      message.success("Voice chat connected");
    } catch (e: any) {
      message.error(e?.message || "Connect failed");
      await cleanupLocal({ stopRemote: true });
    }
  };

  const onDisconnect = async () => {
    await cleanupLocal({ stopRemote: true });
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

  // Auto-scroll to bottom when logs change
  useEffect(() => {
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [logs]);

  useEffect(() => {
    // defaults
    const stableUserId = getOrCreateStableId("fina_demo_voice_user_id");
    const stableRoomId = localStorage.getItem("fina_demo_voice_room_id") || newId();
    form.setFieldsValue({
      roomId: stableRoomId,
      userId: stableUserId,
      taskId: "voice_agent",
    });

    return () => {
      void cleanupLocal({ stopRemote: true });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: "100%" }} size={16}>
        <div>
          <Space style={{ width: "100%", justifyContent: "space-between", alignItems: "center" }}>
            <Title level={3} style={{ margin: 0 }}>
              Voice Agent
            </Title>
            <Space>
              <Space>
                <span>Status</span>
                {statusTag}
              </Space>
              <Button
                type={status === "connected" ? "default" : "primary"}
                danger={status === "connected"}
                onClick={() => {
                  if (status === "connected") {
                    onDisconnect();
                  } else {
                    setIsConnectModalVisible(true);
                  }
                }}
                disabled={status === "connecting"}
              >
                {status === "connected" ? "Disconnect" : "Connect"}
              </Button>
              <Button onClick={onToggleMic} disabled={status !== "connected"}>
                {audioEnabled ? "Mute" : "Unmute"}
              </Button>
            </Space>
          </Space>
        </div>
        {autoplayBlockedUserId !== "" && (
          <Alert
            type="warning"
            showIcon
            message="Audio playback blocked"
            description={
              <Space direction="vertical">
                <div>Your browser blocked audio autoplay. Click the button to resume playback.</div>
                <Button onClick={() => resumePlay(autoplayBlockedUserId || undefined)}>Resume Playback</Button>
              </Space>
            }
          />
        )}
        <div
          ref={logsContainerRef}
        >
          {messageListRef.current.length === 0 ? (
            <Text type="secondary">No messages yet.</Text>
          ) : (
            messageListRef.current.map((l) => {
              const isAssistantMessage = l.userId.startsWith("voice_chat_");
              const isCurrentUser = !isAssistantMessage && l.userId !== "system";

              return (
                <Bubble
                  key={l.id}
                  content={
                    <div>
                      {l.userId === "system" && (
                        <div style={{ fontSize: "12px", color: "#909399", marginBottom: "4px" }}>
                          {l.userId}
                        </div>
                      )}
                      <div style={{ fontSize: "14px", lineHeight: "1.4" }}>
                        {l.text}
                      </div>
                    </div>
                  }
                  placement={isCurrentUser ? "end" : "start"}
                  // footer={
                  //   <div style={{
                  //     fontSize: "11px",
                  //     color: "#c0c4cc",
                  //     textAlign: "right"
                  //   }}>
                  //     {new Date(l.ts).toLocaleTimeString()}
                  //   </div>
                  // }
                  styles={{
                    content: {
                      background: isCurrentUser ? "linear-gradient(1777deg, rgba(153, 164, 255, 0.38), rgba(231, 243, 248, 0.38) 27%)" : "transparent",
                    }
                  }}
                />
              );
            })
          )}
        </div>
      </Space>

      <Modal
        title="Connect to Voice Agent"
        open={isConnectModalVisible}
        onOk={() => {
          if (botIdInput.trim()) {
            setIsConnectModalVisible(false);
            onConnect(botIdInput.trim());
            setBotIdInput("");
          } else {
            message.error("Please enter Bot ID");
          }
        }}
        onCancel={() => {
          setIsConnectModalVisible(false);
          setBotIdInput("");
        }}
        okText="Connect"
        cancelText="Cancel"
      >
        <Form layout="vertical">
          <Form.Item label="Bot ID" required>
            <Input
              placeholder="Enter Bot ID"
              value={botIdInput}
              onChange={(e) => setBotIdInput(e.target.value)}
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};
