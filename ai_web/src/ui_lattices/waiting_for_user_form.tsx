import {
  regsiterElement,
  ElementMeta,
  ElementProps,
} from "@axiom-lattice/react-sdk";
import { FC } from "react";
import { RefreshStateUI } from "./refresh_state";

type ElementPropsWithHandler<T = any> = ElementProps<T> & { eventHandler?: any };

const WaitingForUserFormUI: FC<ElementPropsWithHandler> = ({ eventHandler }) => {
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
