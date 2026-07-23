import { Tag } from "antd";
import type { ComponentProps, CSSProperties } from "react";

const overflowSafeStyle: CSSProperties = {
  maxWidth: "100%",
  height: "auto",
  whiteSpace: "normal",
  overflowWrap: "anywhere",
  wordBreak: "break-word",
  lineHeight: "20px",
  verticalAlign: "top",
};

export function OverflowSafeTag({ style, ...props }: ComponentProps<typeof Tag>) {
  return <Tag {...props} style={{ ...style, ...overflowSafeStyle }} />;
}
