import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DataMine AI — Training Data Pipeline",
  description:
    "Automated end-to-end data mining and curation for AI training datasets.",
  icons: {
    icon: [
      { url: "/icon.svg", type: "image/svg+xml" },
    ],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/icon.svg" type="image/svg+xml" />
      </head>
      <body className="bg-gray-50 text-gray-900 antialiased">{children}</body>
    </html>
  );
}
