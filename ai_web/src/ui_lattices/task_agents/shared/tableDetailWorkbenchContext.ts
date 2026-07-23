import { createContext, useContext } from "react";

export interface TableDetailWorkbenchContextValue {
  detailExpanded: boolean;
  setDetailExpanded: (expanded: boolean) => void;
}

export const TableDetailWorkbenchContext = createContext<TableDetailWorkbenchContextValue>({
  detailExpanded: false,
  setDetailExpanded: () => undefined,
});

export function useTableDetailWorkbenchContext() {
  return useContext(TableDetailWorkbenchContext);
}
