import { Database } from "lucide-react";

export function Header() {
  return (
    <header className="border-b border-border-color bg-bg-secondary/80 backdrop-blur-md sticky top-0 z-50 animate-slide-in-left">
      <div className="flex items-center px-6 py-4">
        <div className="flex items-center gap-3 group">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-accent-primary to-blue-500 flex items-center justify-center shadow-lg transition-all duration-300 group-hover:scale-110 group-hover:rotate-3">
            <Database className="w-6 h-6 text-bg-primary transition-transform duration-300 group-hover:scale-110" />
          </div>
          <div>
            <h1 className="glow-text transition-all duration-300 group-hover:tracking-wide" style={{ color: 'var(--text-primary)' }}>Explainable DBMS</h1>
          </div>
        </div>
      </div>
    </header>
  );
}
