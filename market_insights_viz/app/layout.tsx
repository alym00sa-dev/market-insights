import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AI in Education Evidence Dashboard",
  description: "Strategic framework for navigating research evidence and investment priorities across AI-enabled educational interventions",
  icons: {
    icon: '/brain-icon.svg',
  },
  openGraph: {
    title: "AI in Education Evidence Dashboard",
    description: "Strategic framework for navigating research evidence and investment priorities across AI-enabled educational interventions",
    type: "website",
    siteName: "AI in Education Evidence Dashboard",
  },
  twitter: {
    card: "summary_large_image",
    title: "AI in Education Evidence Dashboard",
    description: "Strategic framework for navigating research evidence and investment priorities across AI-enabled educational interventions",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
