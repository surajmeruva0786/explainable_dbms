import { useState } from "react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { Label } from "./ui/label";
import { BarChart3 } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface QuestionAnswerSectionProps {
  hasAnalysis: boolean;
  analysisId: string;
}

const explanationData = [
  { name: 'Feature 1', importance: 0.85 },
  { name: 'Feature 2', importance: 0.65 },
  { name: 'Feature 3', importance: 0.45 },
  { name: 'Feature 4', importance: 0.30 },
  { name: 'Feature 5', importance: 0.15 },
];

export function QuestionAnswerSection({ hasAnalysis, analysisId }: { hasAnalysis: boolean; analysisId: string }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [plotUrl, setPlotUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleAsk = async () => {
    if (!question.trim() || !hasAnalysis) return;

    setIsLoading(true);
    setAnswer("");
    setPlotUrl(null);

    try {
      const response = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: question,
          analysis_id: analysisId,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get answer');
      }

      const data = await response.json();
      setAnswer(data.answer);
      if (data.plot_url) {
        setPlotUrl(data.plot_url);
      }
    } catch (error) {
      console.error("Query error:", error);
      setAnswer("Sorry, I encountered an error while processing your question.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="border-t border-border-color bg-bg-secondary/80 backdrop-blur-sm">
      <div className="grid grid-cols-2 gap-4 p-6">
        {/* Ask a question */}
        <div className="space-y-3 animate-fade-in" style={{ animationDelay: '0.1s', opacity: 0 }}>
          <Label style={{ color: 'var(--text-primary)' }}>
            Ask a question
          </Label>
          <Textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What insights can you provide about this data?"
            className="min-h-[100px] bg-bg-primary border-border-color focus:border-accent-primary focus:ring-2 focus:ring-accent-primary/20 resize-none transition-all duration-300 hover:shadow-lg"
            style={{ color: 'var(--text-primary)' }}
            disabled={!hasAnalysis || isLoading}
          />
          <Button
            onClick={handleAsk}
            disabled={!hasAnalysis || !question.trim() || isLoading}
            className="w-full bg-text-secondary hover:bg-text-secondary/90 text-bg-primary transform hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 shadow-lg hover:shadow-xl"
          >
            {isLoading ? "Analyzing..." : "Ask"}
          </Button>
        </div>

        {/* Answer */}
        <div className="space-y-3 animate-fade-in" style={{ animationDelay: '0.2s', opacity: 0 }}>
          <Label style={{ color: 'var(--text-primary)' }}>
            Answer
          </Label>
          <Textarea
            value={answer}
            readOnly
            placeholder="Answer will appear here..."
            className="min-h-[100px] bg-bg-primary border-border-color resize-none transition-all duration-300 font-mono text-sm leading-relaxed"
            style={{ color: 'var(--text-primary)' }}
          />
        </div>
      </div>

      {/* Explanation Plot */}
      {plotUrl && (
        <div className="px-6 pb-6 animate-scale-in">
          <div className="bg-bg-primary rounded-lg border border-border-color p-4 hover:border-accent-primary/40 transition-all duration-300 hover:shadow-lg">
            <div className="flex items-center gap-2 mb-4 group">
              <BarChart3 className="w-4 h-4 transition-transform duration-300 group-hover:scale-110" style={{ color: 'var(--accent-primary)' }} />
              <h3 className="text-sm" style={{ color: 'var(--text-primary)' }}>Explanation Plot</h3>
            </div>
            <div className="h-[300px] flex items-center justify-center bg-white/5 rounded-md overflow-hidden">
              <img
                src={`http://localhost:8000${plotUrl}`}
                alt="Explanation Plot"
                className="max-h-full max-w-full object-contain"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
