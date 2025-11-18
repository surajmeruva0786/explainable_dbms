import { X, BarChart3, LineChart, PieChart, TrendingUp, Database } from "lucide-react";
import { BarChart, Bar, LineChart as RechartsLine, Line, PieChart as RechartsPie, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

interface AnalysisVisualizationsProps {
  hasData: boolean;
  showError?: boolean;
}

// Mock data for charts
const barData = [
  { name: 'Jan', value: 4000 },
  { name: 'Feb', value: 3000 },
  { name: 'Mar', value: 2000 },
  { name: 'Apr', value: 2780 },
  { name: 'May', value: 1890 },
  { name: 'Jun', value: 2390 },
];

const lineData = [
  { name: 'Week 1', value: 2400 },
  { name: 'Week 2', value: 1398 },
  { name: 'Week 3', value: 9800 },
  { name: 'Week 4', value: 3908 },
];

const pieData = [
  { name: 'Category A', value: 400 },
  { name: 'Category B', value: 300 },
  { name: 'Category C', value: 300 },
  { name: 'Category D', value: 200 },
];

const COLORS = ['#00F2EA', '#3b82f6', '#8b5cf6', '#ec4899'];

export function AnalysisVisualizations({ hasData, showError }: AnalysisVisualizationsProps) {
  const PlotContainer = ({ children, title, icon: Icon }: { children: React.ReactNode; title: string; icon: any }) => (
    <div className="bg-bg-primary rounded-lg border border-border-color p-4 hover:border-accent-primary/40 transition-all duration-300 transform hover:scale-[1.02] hover:shadow-lg group">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 transition-all duration-300 group-hover:scale-110 group-hover:rotate-12" style={{ color: 'var(--accent-primary)' }} />
          <h3 className="text-sm" style={{ color: 'var(--text-primary)' }}></h3>
        </div>
        <X className="w-4 h-4 cursor-pointer hover:text-accent-primary hover:rotate-90 transition-all duration-300" style={{ color: 'var(--text-secondary)' }} />
      </div>
      <div className="h-[250px] flex items-center justify-center">
        {showError ? (
          <p className="text-xs px-3 py-2 rounded-full bg-red-500/20 border border-red-500 animate-scale-in" style={{ color: '#ef4444' }}>
            Error
          </p>
        ) : !hasData ? (
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            No data to display
          </p>
        ) : (
          children
        )}
      </div>
    </div>
  );

  return (
    <div className="flex-1 p-6 space-y-4 animate-slide-in-right">
      <div className="flex items-center justify-between mb-4">
        <h2 className="animate-fade-in" style={{ color: 'var(--text-primary)' }}>Analysis Visualizations</h2>
      </div>

      {/* Top Row - 2 plots */}
      <div className="grid grid-cols-2 gap-4 animate-fade-in" style={{ animationDelay: '0.2s', opacity: 0 }}>
        <PlotContainer title="Sales Trend" icon={LineChart}>
          <ResponsiveContainer width="100%" height="100%">
            <RechartsLine data={lineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
              <XAxis dataKey="name" stroke="var(--text-secondary)" />
              <YAxis stroke="var(--text-secondary)" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'var(--bg-secondary)', 
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)'
                }} 
              />
              <Line type="monotone" dataKey="value" stroke="var(--accent-primary)" strokeWidth={2} animationDuration={800} />
            </RechartsLine>
          </ResponsiveContainer>
        </PlotContainer>

        <PlotContainer title="Monthly Revenue" icon={BarChart3}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
              <XAxis dataKey="name" stroke="var(--text-secondary)" />
              <YAxis stroke="var(--text-secondary)" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'var(--bg-secondary)', 
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)'
                }} 
              />
              <Bar dataKey="value" fill="var(--accent-primary)" animationDuration={800} />
            </BarChart>
          </ResponsiveContainer>
        </PlotContainer>
      </div>

      {/* Bottom Row - 3 plots */}
      <div className="grid grid-cols-3 gap-4 animate-fade-in" style={{ animationDelay: '0.3s', opacity: 0 }}>
        <PlotContainer title="Distribution" icon={PieChart}>
          <ResponsiveContainer width="100%" height="100%">
            <RechartsPie>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
                animationDuration={800}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'var(--bg-secondary)', 
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)'
                }} 
              />
            </RechartsPie>
          </ResponsiveContainer>
        </PlotContainer>

        <PlotContainer title="Growth Rate" icon={TrendingUp}>
          <ResponsiveContainer width="100%" height="100%">
            <RechartsLine data={lineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
              <XAxis dataKey="name" stroke="var(--text-secondary)" />
              <YAxis stroke="var(--text-secondary)" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'var(--bg-secondary)', 
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)'
                }} 
              />
              <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} animationDuration={800} />
            </RechartsLine>
          </ResponsiveContainer>
        </PlotContainer>

        <PlotContainer title="Data Overview" icon={Database}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
              <XAxis dataKey="name" stroke="var(--text-secondary)" />
              <YAxis stroke="var(--text-secondary)" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'var(--bg-secondary)', 
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  color: 'var(--text-primary)'
                }} 
              />
              <Bar dataKey="value" fill="#8b5cf6" animationDuration={800} />
            </BarChart>
          </ResponsiveContainer>
        </PlotContainer>
      </div>
    </div>
  );
}
