import {
  regsiterElement,
  ElementMeta,
  ElementProps,
} from "@axiom-lattice/react-sdk";
import { FC } from "react";
import { RefreshStateUI } from "./refresh_state";

type ElementPropsWithHandler<T = any> = ElementProps<T> & { eventHandler?: any };

const WaitingForOutboundCallUI: FC<ElementPropsWithHandler> = ({ eventHandler }) => {
  return (
    <div>
      <RefreshStateUI
        eventHandler={eventHandler}
        component_key="waiting_for_outbound_call"
        data={{}}
      />
    </div>
  );
};

const WaitingForOutboundCallMeta: ElementMeta = {
  card_view: WaitingForOutboundCallUI,
};

regsiterElement("waiting_for_outbound_call", WaitingForOutboundCallMeta);
