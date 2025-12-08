import { useState } from "react";
import { Header } from "./components/Header";
import { DBMSControlSidebar, AnalysisParams } from "./components/DBMSControlSidebar";
import { AnalysisVisualizations } from "./components/AnalysisVisualizations";
import { QuestionAnswerSection } from "./components/QuestionAnswerSection";
import { Footer } from "./components/Footer";

interface AnalysisResult {
  message: string;
  analysis_id: string;
  plots: Record<string, string>;
}

export default function App() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [hasAnalysis, setHasAnalysis] = useState(false);
  const [showError, setShowError] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);

  const handleRunAnalysis = async (params: AnalysisParams) => {
    if (!params.csvFile) {
      setShowError(true);
      setErrorMessage("Please upload a CSV file");
      return;
    }

    setIsAnalyzing(true);
    setShowError(false);
    setErrorMessage("");

    try {
      // Call the backend /api/analyze endpoint
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          filename: params.csvFile.name,
          target_column: params.targetColumn,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed');
      }

      const result: AnalysisResult = await response.json();

      setAnalysisResult(result);
      setHasAnalysis(true);
      console.log('Analysis complete:', result);

    } catch (error) {
      console.error('Analysis error:', error);
      setShowError(true);
      setErrorMessage(error instanceof Error ? error.message : 'Failed to run analysis');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="h-screen flex flex-col dark">
      <Header />

      <div className="flex-1 flex overflow-hidden">
        <DBMSControlSidebar
          onRunAnalysis={handleRunAnalysis}
          isAnalyzing={isAnalyzing}
        />

        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto">
            <AnalysisVisualizations
              hasData={hasAnalysis}
              showError={showError}
              errorMessage={errorMessage}
              analysisResult={analysisResult}
            />
          </div>

          <QuestionAnswerSection
            hasAnalysis={hasAnalysis}
            analysisId={analysisResult?.analysis_id || ""}
          />
        </div>
      </div>

      <Footer />
    </div>
  );
}
