import { LatticeChatShell } from "@axiom-lattice/react-sdk";
import { Conversations, Prompts } from "@ant-design/x";
import { createStyles } from "antd-style";
import React, {
  useCallback,
  useEffect,
  useState,
  useImperativeHandle,
  forwardRef,
} from "react";
import { Chating } from "@axiom-lattice/react-sdk";
import { useGetIdentity } from "@refinedev/core";
import { type GetProp } from "antd";
import { useChat } from "@axiom-lattice/react-sdk";
import { SideAppViewBrowser } from "@axiom-lattice/react-sdk";
import { AxiomLatticeProvider } from "@axiom-lattice/react-sdk";
import { TOKEN_KEY } from "../../authProvider";

// Keep chat API base configurable (backend in this repo is mounted under `/bff`).
// In production, VITE_API_URL should be set to /api, so this becomes /api/bff
const apiUrl = (import.meta.env.VITE_API_URL as string | undefined) 
  ? `${import.meta.env.VITE_API_URL}/bff`
  : "http://localhost:6203/bff";

// Utility function to concatenate class names
const cn = (...classNames: Array<string | undefined | false>) =>
  classNames.filter(Boolean).join(" ");

const defaultConversationsItems: GetProp<typeof Conversations, "items"> = [];

const useStyle = createStyles(({ token, css }) => {
  return {
    layout: css`
      width: 100%;
      /* min-width: 1000px; */
      height: 100%;
      border-radius: ${token.borderRadius}px;
      display: flex;
      background: ${token.colorBgLayout}95;
      font-family: ${token.fontFamily}, sans-serif;
      position: relative;
      overflow: hidden;
      padding: 16px;
      padding-top: 2px;
      gap: 16px;

      .ant-prompts {
        color: ${token.colorText};
      }
    `,
    menu: css`
      background: ${token.colorBgContainer}90;
      width: 280px;
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
      transition: all 0.3s ease;
      overflow: hidden;
      position: relative;
      border-radius: ${token.borderRadiusLG}px;
      box-shadow: ${token.boxShadow};

      &.open {
        background: transparent;
        box-shadow: none;
        margin-left: -16px;
        height: 100%;
      }

      &.collapsed {
        width: 64px;
        height: fit-content;
        border-radius: 32px;
        .ant-conversations {
          width: 64px;
        }

        .ant-conversations-list {
          display: none !important;
        }

        .btn-text {
          display: none !important;
        }
      }
    `,
    menuToggle: css`
      position: relative;
      bottom: 20px;
      left: 24px;
      z-index: 1;

      &.collapsed {
        left: 16px;
      }
    `,
    logoutBtn: css`
      position: absolute;
      bottom: 32px;
      left: 24px;
      z-index: 1;
      color: ${token.colorTextSecondary};

      &:hover {
        color: ${token.colorError};
      }

      &.collapsed {
        left: 16px;
      }

      .btn-text {
        margin-left: 8px;
        transition: opacity 0.3s ease, width 0.3s ease;
      }

      &.collapsed .btn-text {
        display: none;
      }
    `,
    conversations: css`
      padding: 0 12px;
      flex: 1;
      overflow-y: auto;
      transition: padding 0.3s ease;

      .collapsed & {
        padding: 0 4px;
      }
    `,
    mainContent: css`
      min-width: 320px;
      flex: 1;
      display: flex;
      position: relative;
      overflow: hidden;
      gap: 16px;
      justify-content: center;
      &.open {
        box-shadow: ${token.boxShadow};
        background: ${token.colorBgContainer}90;
        margin-left: 0px;
      }
      border-radius: ${token.borderRadiusLG}px;
      .ant-bubble-content {
        .ant-card {
          background: #ffffff4f;
        }
      }

      pre {
        white-space: pre-wrap;
        word-break: break-all;
      }
    `,
    chat: css`
      width: 100%;
      max-width: 1000px;
      box-sizing: border-box;
      display: flex;
      flex-direction: column;
      padding: ${token.paddingXS}px;
      gap: 0px;
      transition: all 0.3s ease;
      flex-shrink: 0;

      &.drawer-open {
        min-width: 200px;
        max-width: 466px;
      }
    `,
    detailPanel: css`
      display: flex;
      flex-direction: column;
      width: 0;
      background: ${token.colorBgContainer}90;

      transition: all 0.3s ease;
      overflow: hidden;
      flex-shrink: 0;
      border-radius: ${token.borderRadiusLG}px;

      &.open {
        width: 66%;
        box-shadow: ${token.boxShadow};
      }

      &.small {
        width: 22%;
      }

      &.middle {
        width: 44%;
      }

      &.large {
        width: 66%;
      }
      &.full {
        width: 98%;
        position: absolute;
        background-color: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        //border: 1px solid rgba(255, 255, 255, 0.2);
        z-index: 10;
        bottom: 16px;
        top: 2px;
      }
    `,
    detailContent: css`
      padding: 8px 8px;
      height: 100%;
      flex: 1;
      overflow: hidden;
      pre {
        white-space: pre-wrap;
        word-break: break-all;
      }
    `,
    detailHeader: css`
      padding: 16px 24px;
      border-bottom: 1px solid ${token.colorBorder};
      display: flex;
      justify-content: space-between;
      align-items: center;

      h3 {
        margin: 0;
        font-size: 16px;
        font-weight: 500;
      }
    `,
    messages: css`
      padding: 0 20px;
      flex: 1;
    `,
    placeholder: css`
      padding-top: 32px;
    `,
    sender: css`
      box-shadow: ${token.boxShadow};
    `,
    logo: css`
      display: flex;
      height: 72px;
      align-items: center;
      justify-content: start;
      padding: 0 24px;
      box-sizing: border-box;
      white-space: nowrap;
      overflow: hidden;
      transition: padding 0.3s ease;

      img {
        width: 24px;
        height: 24px;
        display: inline-block;
      }

      span {
        display: inline-block;
        margin: 0 8px;
        font-weight: bold;
        color: ${token.colorText};
        font-size: 16px;
        opacity: 1;
        transition: opacity 0.3s ease, width 0.3s ease, margin 0.3s ease;
      }

      &.collapsed {
        padding: 0 20px;
        justify-content: center;

        span {
          opacity: 0;
          width: 0;
          margin: 0;
        }
      }
    `,
    addBtn: css`
      background: #1677ff0f;
      border: 1px solid #1677ff34;
      width: calc(100% - 24px);
      margin: 0 12px 24px 12px;
      white-space: nowrap;
      overflow: hidden;
      transition: all 0.3s ease;

      &.collapsed {
        width: 48px;
        min-width: 48px;
        margin: 0 auto 24px auto;
        justify-content: center;
        padding: 0;
      }
    `,
  };
});

// Expose sendMessage method via ref
export interface ChatBotRef {
  sendMessage: (message: string) => void;
}

export const ChatBotCompnent = forwardRef<
  ChatBotRef,
  {
    id: string;
    threadId: string;
    name: string;
  }
>(({ id, threadId, name }, ref) => {
  const token = localStorage.getItem(TOKEN_KEY);
  const headers = token ? { Authorization: `Bearer ${token}` } : undefined;

  return (
    <LatticeChatShell
      initialConfig={{
        baseURL: apiUrl,
        headers,
      }}
    />
  );
});
