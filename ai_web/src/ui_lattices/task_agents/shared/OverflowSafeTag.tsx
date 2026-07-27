import { Tag } from "antd";
import type { ComponentProps, CSSProperties } from "react";

const overflowSafeStyle: CSSProperties = {
  boxSizing: "border-box",
  maxWidth: "100%",
  minWidth: 0,
  height: "auto",
  marginInlineEnd: 0,
  whiteSpace: "normal",
  overflowWrap: "anywhere",
  wordBreak: "break-word",
  lineHeight: "20px",
  verticalAlign: "top",
};

export function OverflowSafeTag({ style, ...props }: ComponentProps<typeof Tag>) {
  return <Tag {...props} style={{ ...style, ...overflowSafeStyle }} />;
}
