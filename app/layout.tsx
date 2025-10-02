import "./globals.css";


export const metadata = {
  title: "NFPA 13 / 13R – Q&A",
  description: "Local RAG frontend for NFPA 13-2022 & PCI NFPA 13R",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <div className="max-w-3xl mx-auto p-4 md:p-6">{children}</div>
      </body>
    </html>
  );
}
