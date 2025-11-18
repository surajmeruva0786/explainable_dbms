import { useState } from "react";
import { Download, Maximize2, Copy, Check } from "lucide-react";
import { Button } from "./ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { ImageWithFallback } from "./figma/ImageWithFallback";

interface OutputAreaProps {
  generatedImage?: string;
  generatedText?: string;
  isGenerating: boolean;
}

export function OutputArea({ generatedImage, generatedText, isGenerating }: OutputAreaProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (generatedText) {
      navigator.clipboard.writeText(generatedText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="flex-1 p-6 overflow-y-auto">
      <div className="max-w-5xl mx-auto space-y-6">
        <div>
          <h2 className="mb-2" style={{ color: 'var(--text-primary)' }}>Output</h2>
          <p style={{ color: 'var(--text-secondary)' }}>
            Generated results will appear here
          </p>
        </div>

        <Tabs defaultValue="image" className="w-full">
          <TabsList className="bg-bg-secondary border border-border-color">
            <TabsTrigger 
              value="image"
              className="data-[state=active]:bg-accent-primary data-[state=active]:text-bg-primary"
            >
              Image Output
            </TabsTrigger>
            <TabsTrigger 
              value="text"
              className="data-[state=active]:bg-accent-primary data-[state=active]:text-bg-primary"
            >
              Text Output
            </TabsTrigger>
          </TabsList>

          <TabsContent value="image" className="mt-6">
            <div className="relative group">
              {isGenerating ? (
                <div className="aspect-video bg-bg-secondary rounded-lg border border-border-color flex items-center justify-center">
                  <div className="space-y-4 text-center">
                    <div className="relative w-20 h-20 mx-auto">
                      {/* Futuristic loading animation */}
                      <div className="absolute inset-0 border-4 border-transparent border-t-accent-primary rounded-full animate-spin"></div>
                      <div className="absolute inset-2 border-4 border-transparent border-t-blue-500 rounded-full animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }}></div>
                      <div className="absolute inset-4 border-4 border-transparent border-t-purple-500 rounded-full animate-spin" style={{ animationDuration: '2s' }}></div>
                    </div>
                    <p style={{ color: 'var(--text-primary)' }}>Generating your image...</p>
                    <div className="flex items-center justify-center gap-1">
                      <div className="w-2 h-2 bg-accent-primary rounded-full animate-pulse"></div>
                      <div className="w-2 h-2 bg-accent-primary rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                      <div className="w-2 h-2 bg-accent-primary rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                    </div>
                  </div>
                </div>
              ) : generatedImage ? (
                <>
                  <div className="relative rounded-lg overflow-hidden border border-border-color">
                    <ImageWithFallback
                      src={generatedImage}
                      alt="Generated image"
                      className="w-full h-auto"
                    />
                    
                    {/* Hover overlay */}
                    <div className="absolute inset-0 bg-bg-primary/80 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center gap-2">
                      <Button
                        size="icon"
                        className="bg-accent-primary hover:bg-accent-primary/90 text-bg-primary"
                      >
                        <Download className="w-5 h-5" />
                      </Button>
                      <Button
                        size="icon"
                        className="bg-accent-primary hover:bg-accent-primary/90 text-bg-primary"
                      >
                        <Maximize2 className="w-5 h-5" />
                      </Button>
                    </div>
                  </div>
                </>
              ) : (
                <div className="aspect-video bg-bg-secondary rounded-lg border border-dashed border-border-color flex items-center justify-center">
                  <p style={{ color: 'var(--text-secondary)' }}>
                    No image generated yet. Configure parameters and click Generate.
                  </p>
                </div>
              )}
            </div>

            {generatedImage && (
              <div className="mt-4 p-4 bg-bg-secondary rounded-lg border border-border-color">
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <p style={{ color: 'var(--text-secondary)' }}>Resolution</p>
                    <p style={{ color: 'var(--text-primary)' }}>1024 x 1024</p>
                  </div>
                  <div>
                    <p style={{ color: 'var(--text-secondary)' }}>Format</p>
                    <p style={{ color: 'var(--text-primary)' }}>PNG</p>
                  </div>
                  <div>
                    <p style={{ color: 'var(--text-secondary)' }}>Size</p>
                    <p style={{ color: 'var(--text-primary)' }}>2.4 MB</p>
                  </div>
                </div>
              </div>
            )}
          </TabsContent>

          <TabsContent value="text" className="mt-6">
            <div className="relative">
              {isGenerating ? (
                <div className="p-6 bg-bg-secondary rounded-lg border border-border-color">
                  <div className="space-y-3">
                    <div className="h-4 bg-border-color rounded animate-pulse w-3/4"></div>
                    <div className="h-4 bg-border-color rounded animate-pulse w-full"></div>
                    <div className="h-4 bg-border-color rounded animate-pulse w-5/6"></div>
                  </div>
                </div>
              ) : generatedText ? (
                <div className="relative group">
                  <pre className="p-6 bg-bg-secondary rounded-lg border border-border-color overflow-x-auto font-mono text-sm" style={{ color: 'var(--text-primary)' }}>
                    {generatedText}
                  </pre>
                  <Button
                    onClick={handleCopy}
                    size="icon"
                    className="absolute top-4 right-4 bg-accent-primary hover:bg-accent-primary/90 text-bg-primary opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    {copied ? (
                      <Check className="w-4 h-4" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </Button>
                </div>
              ) : (
                <div className="p-12 bg-bg-secondary rounded-lg border border-dashed border-border-color text-center">
                  <p style={{ color: 'var(--text-secondary)' }}>
                    No text output yet. Configure parameters and click Generate.
                  </p>
                </div>
              )}
            </div>

            {generatedText && (
              <div className="mt-4 p-4 bg-bg-secondary rounded-lg border border-border-color">
                <div className="flex items-center justify-between text-sm">
                  <div>
                    <p style={{ color: 'var(--text-secondary)' }}>Characters</p>
                    <p style={{ color: 'var(--text-primary)' }}>{generatedText.length}</p>
                  </div>
                  <div>
                    <p style={{ color: 'var(--text-secondary)' }}>Words</p>
                    <p style={{ color: 'var(--text-primary)' }}>{generatedText.split(/\s+/).length}</p>
                  </div>
                  <div>
                    <p style={{ color: 'var(--text-secondary)' }}>Lines</p>
                    <p style={{ color: 'var(--text-primary)' }}>{generatedText.split('\n').length}</p>
                  </div>
                </div>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
