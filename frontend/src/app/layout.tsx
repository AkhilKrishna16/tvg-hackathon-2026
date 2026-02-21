import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title:       "Grid Placement Optimizer · Austin TX",
  description: "AI-powered EV infrastructure substation placement optimizer for Austin, Texas.",
};

export const viewport: Viewport = {
  width:        "device-width",
  initialScale:  1,
  themeColor:   "#080812",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full overflow-hidden">
      <body className="h-full overflow-hidden bg-[#080812]">
        {children}
      </body>
    </html>
  );
}
