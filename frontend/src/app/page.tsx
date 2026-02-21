import dynamic from "next/dynamic";

// Dashboard is a full WebGL client component — disable SSR entirely
const Dashboard = dynamic(() => import("@/components/Dashboard"), { ssr: false });

export default function Page() {
  return <Dashboard />;
}
