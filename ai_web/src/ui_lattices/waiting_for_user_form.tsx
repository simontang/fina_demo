import {
  regsiterElement,
  ElementMeta,
  ElementProps,
} from "@axiom-lattice/react-sdk";
import { Button } from "antd";
import { FC } from "react";
import { RefreshStateUI } from "./refresh_state";

const WaitingForUserFormUI: FC<ElementProps> = ({ eventHandler }) => {
  return (
    <div>
      <RefreshStateUI
        eventHandler={eventHandler}
        component_key="waiting_for_user_form"
        data={{}}
      />
    </div>
  );
};

const WaitingForUserFormMeta: ElementMeta = {
  card_view: WaitingForUserFormUI,
};

regsiterElement("waiting_for_user_form", WaitingForUserFormMeta);
