import { Outlet } from "react-router";
import { AppShell } from "./AppShell";
import { CommandPalette } from "./CommandPalette";

export function RootLayout() {
  return (
    <>
      <CommandPalette />
      <AppShell>
        <Outlet />
      </AppShell>
    </>
  );
}
