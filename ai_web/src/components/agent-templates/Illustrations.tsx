import type { CSSProperties, ReactNode } from "react";
import type { IllustrationKey } from "./templateData";

const svgStyle: CSSProperties = {
  width: "100%",
  height: "100%",
  display: "block",
};

function Blob({
  id,
  accent,
  opacity = 0.18,
}: {
  id: string;
  accent: string;
  opacity?: number;
}) {
  return (
    <>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor={accent} stopOpacity={opacity} />
          <stop offset="1" stopColor="#ffffff" stopOpacity={0} />
        </linearGradient>
        <filter id={`${id}-blur`} x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="8" />
        </filter>
      </defs>
      <path
        d="M20,36 C22,12 58,6 78,16 C98,26 128,10 140,30 C152,50 148,80 122,86 C96,92 88,78 66,86 C44,94 18,84 18,62 C18,52 19,44 20,36 Z"
        fill={`url(#${id})`}
        filter={`url(#${id}-blur)`}
      />
    </>
  );
}

function Frame({
  children,
  accent,
  id,
}: {
  children: ReactNode;
  accent: string;
  id: string;
}) {
  return (
    <svg viewBox="0 0 160 96" style={svgStyle} aria-hidden="true">
      <rect x="0" y="0" width="160" height="96" rx="14" fill="#ffffff" />
      <Blob id={id} accent={accent} />
      <g>{children}</g>
    </svg>
  );
}

function stroke(accent: string) {
  return { stroke: accent, strokeWidth: 2, fill: "none", strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
}

function mutedStroke() {
  return { stroke: "rgba(48,49,51,0.22)", strokeWidth: 2, fill: "none", strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
}

export function getIllustration(key: IllustrationKey, accent: string): ReactNode {
  switch (key) {
    case "ebr":
      return (
        <Frame accent={accent} id="tmpl-ebr">
          <rect x="26" y="22" width="108" height="54" rx="10" fill="rgba(255,255,255,0.65)" stroke="rgba(48,49,51,0.10)" />
          <path {...mutedStroke()} d="M40 36 H120" />
          <path {...mutedStroke()} d="M40 48 H106" />
          <path {...mutedStroke()} d="M40 60 H92" />
          <rect x="40" y="64" width="28" height="8" rx="4" fill={accent} opacity="0.85" />
          <path {...stroke(accent)} d="M88 66 L98 56 L108 60 L118 46" />
          <circle cx="118" cy="46" r="3" fill={accent} />
        </Frame>
      );
    case "anomaly":
      return (
        <Frame accent={accent} id="tmpl-anom">
          <circle cx="78" cy="50" r="26" fill="rgba(255,255,255,0.55)" stroke="rgba(48,49,51,0.10)" />
          <path {...mutedStroke()} d="M78 24 V76" />
          <path {...mutedStroke()} d="M52 50 H104" />
          <path {...stroke(accent)} d="M78 50 L96 40" />
          <circle cx="96" cy="40" r="4" fill={accent} />
          <path {...stroke(accent)} d="M118 28 L126 36" />
          <path {...stroke(accent)} d="M126 28 L118 36" />
          <rect x="112" y="22" width="20" height="20" rx="6" fill="rgba(255,255,255,0.65)" stroke="rgba(48,49,51,0.10)" />
        </Frame>
      );
    case "gross_margin":
      return (
        <Frame accent={accent} id="tmpl-gm">
          <rect x="30" y="26" width="100" height="50" rx="10" fill="rgba(255,255,255,0.55)" stroke="rgba(48,49,51,0.10)" />
          <path {...mutedStroke()} d="M42 68 H118" />
          <rect x="46" y="52" width="14" height="16" rx="3" fill={accent} opacity="0.85" />
          <rect x="66" y="44" width="14" height="24" rx="3" fill={accent} opacity="0.65" />
          <rect x="86" y="50" width="14" height="18" rx="3" fill={accent} opacity="0.5" />
          <rect x="106" y="38" width="14" height="30" rx="3" fill={accent} opacity="0.9" />
          <path {...stroke(accent)} d="M46 52 H60 V44 H80 V50 H100 V38 H120" />
        </Frame>
      );
    case "budget":
      return (
        <Frame accent={accent} id="tmpl-budget">
          <path
            d="M44 62 A36 36 0 0 1 116 62"
            stroke="rgba(48,49,51,0.18)"
            strokeWidth="10"
            fill="none"
            strokeLinecap="round"
          />
          <path
            d="M44 62 A36 36 0 0 1 92 30"
            stroke={accent}
            strokeWidth="10"
            fill="none"
            strokeLinecap="round"
          />
          <path {...stroke(accent)} d="M80 62 L100 44" />
          <circle cx="80" cy="62" r="4" fill={accent} />
          <rect x="56" y="66" width="48" height="10" rx="5" fill="rgba(255,255,255,0.55)" stroke="rgba(48,49,51,0.10)" />
          <rect x="58" y="68" width="22" height="6" rx="3" fill={accent} opacity="0.75" />
        </Frame>
      );
    case "funnel":
      return (
        <Frame accent={accent} id="tmpl-funnel">
          <path
            d="M40 28 H120 L96 52 V68 L64 76 V52 Z"
            fill="rgba(255,255,255,0.55)"
            stroke="rgba(48,49,51,0.14)"
            strokeWidth="2"
            strokeLinejoin="round"
          />
          <path {...stroke(accent)} d="M44 34 H116" />
          <path {...stroke(accent)} d="M54 46 H106" opacity="0.9" />
          <path {...stroke(accent)} d="M66 58 H94" opacity="0.75" />
          <circle cx="124" cy="32" r="5" fill={accent} opacity="0.85" />
          <circle cx="130" cy="44" r="3" fill={accent} opacity="0.6" />
        </Frame>
      );
    case "forecast":
      return (
        <Frame accent={accent} id="tmpl-forecast">
          <rect x="30" y="26" width="100" height="50" rx="10" fill="rgba(255,255,255,0.55)" stroke="rgba(48,49,51,0.10)" />
          <path {...mutedStroke()} d="M42 68 H118" />
          <path {...mutedStroke()} d="M42 36 V68" />
          <path {...stroke(accent)} d="M44 62 L62 52 L78 58 L94 44 L112 40" />
          <circle cx="112" cy="40" r="3" fill={accent} />
          <path {...stroke(accent)} d="M112 40 L126 30" opacity="0.6" />
          <path {...stroke(accent)} d="M112 40 L126 40" opacity="0.35" />
        </Frame>
      );
    case "churn":
      return (
        <Frame accent={accent} id="tmpl-churn">
          <circle cx="68" cy="44" r="10" fill="rgba(255,255,255,0.65)" stroke="rgba(48,49,51,0.12)" />
          <path {...mutedStroke()} d="M50 72 C54 60 82 60 86 72" />
          <path {...stroke(accent)} d="M102 32 L120 50" />
          <path {...stroke(accent)} d="M120 32 L102 50" />
          <rect x="96" y="26" width="30" height="30" rx="10" fill="rgba(255,255,255,0.65)" stroke="rgba(48,49,51,0.10)" />
          <path {...stroke(accent)} d="M104 68 H122" />
          <path {...stroke(accent)} d="M104 68 L110 62" />
          <path {...stroke(accent)} d="M104 68 L110 74" />
        </Frame>
      );
    case "segmentation":
      return (
        <Frame accent={accent} id="tmpl-seg">
          <circle cx="78" cy="50" r="26" fill="rgba(255,255,255,0.55)" stroke="rgba(48,49,51,0.12)" />
          <path d="M78 50 L78 24 A26 26 0 0 1 102 40 Z" fill={accent} opacity="0.85" />
          <path d="M78 50 L102 40 A26 26 0 0 1 96 70 Z" fill={accent} opacity="0.55" />
          <path d="M78 50 L96 70 A26 26 0 0 1 56 64 Z" fill={accent} opacity="0.35" />
          <path d="M78 50 L56 64 A26 26 0 0 1 78 24 Z" fill={accent} opacity="0.2" />
          <circle cx="120" cy="32" r="5" fill={accent} opacity="0.65" />
          <circle cx="130" cy="44" r="3" fill={accent} opacity="0.45" />
        </Frame>
      );
    case "inventory":
      return (
        <Frame accent={accent} id="tmpl-inv">
          <rect x="40" y="34" width="30" height="26" rx="6" fill="rgba(255,255,255,0.65)" stroke="rgba(48,49,51,0.14)" />
          <rect x="74" y="28" width="34" height="30" rx="6" fill="rgba(255,255,255,0.65)" stroke="rgba(48,49,51,0.14)" />
          <rect x="60" y="58" width="44" height="22" rx="6" fill="rgba(255,255,255,0.65)" stroke="rgba(48,49,51,0.14)" />
          <path {...stroke(accent)} d="M46 46 H64" opacity="0.9" />
          <path {...stroke(accent)} d="M80 42 H102" opacity="0.75" />
          <path {...stroke(accent)} d="M68 68 H96" opacity="0.6" />
        </Frame>
      );
    case "fulfillment":
      return (
        <Frame accent={accent} id="tmpl-full">
          <rect x="34" y="44" width="64" height="26" rx="8" fill="rgba(255,255,255,0.65)" stroke="rgba(48,49,51,0.14)" />
          <rect x="98" y="52" width="28" height="18" rx="6" fill="rgba(255,255,255,0.65)" stroke="rgba(48,49,51,0.14)" />
          <path {...stroke(accent)} d="M44 56 H76" opacity="0.9" />
          <circle cx="54" cy="72" r="6" fill="#ffffff" stroke={accent} strokeWidth="2" />
          <circle cx="108" cy="72" r="6" fill="#ffffff" stroke={accent} strokeWidth="2" />
          <circle cx="126" cy="34" r="14" fill="rgba(255,255,255,0.65)" stroke="rgba(48,49,51,0.12)" />
          <path {...stroke(accent)} d="M126 34 V26" />
          <path {...stroke(accent)} d="M126 34 L132 38" />
        </Frame>
      );
    default:
      return null;
  }
}
