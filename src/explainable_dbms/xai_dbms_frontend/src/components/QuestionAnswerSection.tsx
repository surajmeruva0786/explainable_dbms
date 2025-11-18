import { useState } from "react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { Label } from "./ui/label";
import { BarChart3 } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface QuestionAnswerSectionProps {
  hasAnalysis: boolean;
}

const explanationData = [
  { name: 'Feature 1', importance: 0.85 },
  { name: 'Feature 2', importance: 0.65 },
  { name: 'Feature 3', importance: 0.45 },
  { name: 'Feature 4', importance: 0.30 },
  { name: 'Feature 5', importance: 0.15 },
];

export function QuestionAnswerSection({ hasAnalysis }: QuestionAnswerSectionProps) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [showExplanation, setShowExplanation] = useState(false);

  const handleAsk = () => {
    if (!question.trim()) return;
    
    // Simulate answer generation
    setAnswer(`Based on the analysis of your dataset:\n\nThe ${question.toLowerCase()} shows a significant correlation with the target variable. The model has identified key patterns in the data that suggest a strong relationship between these features.\n\nKey findings:\n• Primary factor: High correlation coefficient (0.85)\n• Secondary factors: Supporting evidence from multiple features\n• Confidence level: 92%\n\nRecommendation: Further investigation of feature interactions is advised.`);
    setShowExplanation(true);
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
            disabled={!hasAnalysis}
          />
          <Button
            onClick={handleAsk}
            disabled={!hasAnalysis || !question.trim()}
            className="w-full bg-text-secondary hover:bg-text-secondary/90 text-bg-primary transform hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 shadow-lg hover:shadow-xl"
          >
            Ask
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
            className="min-h-[100px] bg-bg-primary border-border-color resize-none transition-all duration-300"
            style={{ color: 'var(--text-primary)' }}
          />
        </div>
      </div>

      {/* Explanation Plot */}
      {showExplanation && (
        <div className="px-6 pb-6 animate-scale-in">
          <div className="bg-bg-primary rounded-lg border border-border-color p-4 hover:border-accent-primary/40 transition-all duration-300 hover:shadow-lg">
            <div className="flex items-center gap-2 mb-4 group">
              <BarChart3 className="w-4 h-4 transition-transform duration-300 group-hover:scale-110" style={{ color: 'var(--accent-primary)' }} />
              <h3 className="text-sm" style={{ color: 'var(--text-primary)' }}>Explanation Plot</h3>
            </div>
            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={explanationData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                  <XAxis type="number" stroke="var(--text-secondary)" />
                  <YAxis type="category" dataKey="name" stroke="var(--text-secondary)" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'var(--bg-secondary)', 
                      border: '1px solid var(--border-color)',
                      borderRadius: '8px',
                      color: 'var(--text-primary)'
                    }} 
                  />
                  <Bar dataKey="importance" fill="var(--accent-primary)" animationDuration={800} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
