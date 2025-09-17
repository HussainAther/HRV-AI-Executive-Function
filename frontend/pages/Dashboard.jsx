import { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import { Chart as ChartJS, LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend } from "chart.js";

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend);

export default function Dashboard() {
  const [dataPoints, setDataPoints] = useState([]);

  useEffect(() => {
    fetch("/api/prediction-log")
      .then((res) => res.json())
      .then((data) => {
        const formatted = data.map((entry) => ({
          time: new Date(entry.timestamp).toLocaleTimeString(),
          score: entry.executive_function_score,
        }));
        setDataPoints(formatted);
      });
  }, []);

  const chartData = {
    labels: dataPoints.map((dp) => dp.time),
    datasets: [{
      label: "Executive Function Score",
      data: dataPoints.map((dp) => dp.score),
      borderColor: "rgb(75, 192, 192)",
      tension: 0.3,
    }],
  };

  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold mb-4">Dashboard</h2>
      <Line data={chartData} />
    </div>
  );
}

