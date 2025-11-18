import { useState } from "react";
import { Header } from "./components/Header";
import { DBMSControlSidebar, AnalysisParams } from "./components/DBMSControlSidebar";
import { AnalysisVisualizations } from "./components/AnalysisVisualizations";
import { QuestionAnswerSection } from "./components/QuestionAnswerSection";
import { Footer } from "./components/Footer";

export default function App() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [hasAnalysis, setHasAnalysis] = useState(false);
  const [showError, setShowError] = useState(false);

  const handleRunAnalysis = (params: AnalysisParams) => {
    setIsAnalyzing(true);
    setShowError(false);
    
    // Simulate analysis process
    setTimeout(() => {
      setIsAnalyzing(false);
      setHasAnalysis(true);
    }, 2000);
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
            />
          </div>
          
          <QuestionAnswerSection hasAnalysis={hasAnalysis} />
        </div>
      </div>
      
      <Footer />
    </div>
  );
}
