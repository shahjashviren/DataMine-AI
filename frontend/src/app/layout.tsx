import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DataMine AI — Training Data Pipeline",
  description:
    "Automated end-to-end data mining and curation for AI training datasets.",
  icons: {
    icon: "/icon.svg",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50 text-gray-900 antialiased">{children}</body>
    </html>
  );
}
