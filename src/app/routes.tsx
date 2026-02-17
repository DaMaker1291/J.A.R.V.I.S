import { createBrowserRouter } from "react-router";
import { RootLayout } from "./components/RootLayout";
import { Dashboard } from "./components/Dashboard";
import { DemoPageDark } from "./components/DemoPageDark";
import { MonitoringPageDark } from "./components/MonitoringPageDark";

// Handle GitHub Pages basename
const basename = '/J.A.R.V.I.S';

export const router = createBrowserRouter([
  {
    path: "/",
    Component: RootLayout,
    children: [
      {
        index: true,
        Component: Dashboard,
      },
      {
        path: "demo",
        Component: DemoPageDark,
      },
      {
        path: "monitoring",
        Component: MonitoringPageDark,
      },
    ],
  },
], {
  basename,
});