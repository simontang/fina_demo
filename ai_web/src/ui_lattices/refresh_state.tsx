import {
  regsiterElement,
  ElementMeta,
  ElementProps,
} from "@axiom-lattice/react-sdk";
import { Button } from "antd";
import { SyncOutlined } from "@ant-design/icons";
import { FC, useState, useEffect, useRef } from "react";

export const RefreshStateUI: FC<ElementProps> = ({ eventHandler }) => {
  const [isMonitoring, setIsMonitoring] = useState(true);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const eventHandlerRef = useRef(eventHandler);

  // Keep eventHandler ref updated
  useEffect(() => {
    eventHandlerRef.current = eventHandler;
  }, [eventHandler]);

  useEffect(() => {
    // Always clear existing interval first to prevent duplicates
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (isMonitoring) {
      // Execute immediately when starting
      eventHandlerRef.current("get_latest_status", {}, "");
      // Then execute every 10 seconds
      intervalRef.current = setInterval(() => {
        eventHandlerRef.current("get_latest_status", {}, "");
      }, 5000);
    }

    // Cleanup on unmount or when isMonitoring changes
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isMonitoring]);

  return (
    <div>
      <Button
        type={isMonitoring ? "primary" : "default"}
        shape="circle"
        icon={<SyncOutlined />}
        onClick={() => setIsMonitoring(!isMonitoring)}
      />
    </div>
  );
};

const RefreshStateMeta: ElementMeta = {
  card_view: RefreshStateUI,
};

regsiterElement("refresh_state", RefreshStateMeta);
