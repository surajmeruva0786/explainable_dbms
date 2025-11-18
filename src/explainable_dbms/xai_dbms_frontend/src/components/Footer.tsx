import { Settings, Code, Sparkles } from "lucide-react";
import { Button } from "./ui/button";

export function Footer() {
  return (
    <footer className="border-t border-border-color bg-bg-secondary/80 backdrop-blur-md">
      <div className="flex items-center justify-center gap-6 px-6 py-3 animate-fade-in">
        <Button 
          variant="ghost" 
          size="sm" 
          className="gap-2 hover:bg-bg-primary text-xs transition-all duration-200 hover:scale-105 group"
          style={{ color: 'var(--text-secondary)' }}
        >
          <Code className="w-3 h-3 transition-transform duration-300 group-hover:rotate-12" />
          Use via API
        </Button>
        
        <div className="h-4 w-px bg-border-color"></div>
        
        <Button 
          variant="ghost" 
          size="sm" 
          className="gap-2 hover:bg-bg-primary text-xs transition-all duration-200 hover:scale-105 group"
          style={{ color: 'var(--text-secondary)' }}
        >
          <Sparkles className="w-3 h-3 transition-transform duration-300 group-hover:scale-125 group-hover:rotate-12" />
          Built with Gradio
        </Button>
        
        <div className="h-4 w-px bg-border-color"></div>
        
        <Button 
          variant="ghost" 
          size="sm" 
          className="gap-2 hover:bg-bg-primary text-xs transition-all duration-200 hover:scale-105 group"
          style={{ color: 'var(--text-secondary)' }}
        >
          <Settings className="w-3 h-3 transition-transform duration-300 group-hover:rotate-90" />
          Settings
        </Button>
      </div>
    </footer>
  );
}
