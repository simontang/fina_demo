import { createPortal } from "react-dom";
import type { RefObject } from "react";
import { useEffect, useRef, useState } from "react";
import TemplateGallery from "./TemplateGallery";

function findNewProjectLabelNode(root: HTMLElement): HTMLElement | null {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const text = (node.nodeValue || "").trim();
      return text === "New Project" ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
    },
  });

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const node = walker.nextNode() as Text | null;
    if (!node) return null;

    const el = node.parentElement;
    if (!el) continue;
    if (el.closest("button")) continue; // avoid the top-right action button
    return el;
  }
}

function isTileLike(el: Element): boolean {
  if (!(el instanceof HTMLElement)) return false;
  const rect = el.getBoundingClientRect();
  return rect.width >= 160 && rect.height >= 120;
}

function findGridContainer(labelEl: HTMLElement): HTMLElement | null {
  let current: HTMLElement | null = labelEl.closest("div");
  while (current && current.parentElement) {
    const parent = current.parentElement;
    const children = Array.from(parent.children);
    const tileLikeCount = children.filter(isTileLike).length;
    if (children.length >= 2 && tileLikeCount >= 2) {
      return current;
    }
    current = parent;
  }
  return null;
}

export default function ProjectTemplatesPortal({
  shellRootRef,
}: {
  shellRootRef: RefObject<HTMLDivElement | null>;
}) {
  const [mountEl, setMountEl] = useState<HTMLDivElement | null>(null);
  const mountRef = useRef<HTMLDivElement | null>(null);
  const observerRef = useRef<MutationObserver | null>(null);

  useEffect(() => {
    const rootEl = shellRootRef.current;
    if (!rootEl) return;

    const ensureMounted = () => {
      const labelEl = findNewProjectLabelNode(rootEl);
      if (!labelEl) {
        if (mountRef.current) {
          mountRef.current.remove();
          mountRef.current = null;
          setMountEl(null);
        }
        return;
      }

      if (mountRef.current) return;

      const gridContainer = findGridContainer(labelEl);
      if (!gridContainer) return;

      const nextMount = document.createElement("div");
      nextMount.style.width = "100%";
      nextMount.setAttribute("data-agent-templates", "true");
      gridContainer.insertAdjacentElement("afterend", nextMount);

      mountRef.current = nextMount;
      setMountEl(nextMount);
    };

    ensureMounted();

    observerRef.current?.disconnect();
    observerRef.current = new MutationObserver(() => {
      ensureMounted();
    });
    observerRef.current.observe(rootEl, { childList: true, subtree: true, characterData: true });

    return () => {
      observerRef.current?.disconnect();
      observerRef.current = null;
      if (mountRef.current) {
        mountRef.current.remove();
        mountRef.current = null;
      }
      setMountEl(null);
    };
  }, [shellRootRef]);

  if (!mountEl) return null;
  return createPortal(<TemplateGallery />, mountEl);
}
