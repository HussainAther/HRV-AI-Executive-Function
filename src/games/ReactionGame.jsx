import React, { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const ReactionGame = () => {
  const [status, setStatus] = useState("waiting"); // waiting, ready, now, result
  const [reactionTime, setReactionTime] = useState(null);
  const [timeoutId, setTimeoutId] = useState(null);
  const [startTime, setStartTime] = useState(null);
  const [log, setLog] = useState([]);

  const startGame = () => {
    setStatus("ready");
    const delay = Math.floor(Math.random() * 3000) + 2000; // 2s–5s
    const id = setTimeout(() => {
      setStatus("now");
      setStartTime(Date.now());
    }, delay);
    setTimeoutId(id);
  };

  const handleClick = () => {
    if (status === "waiting") {
      startGame();
    } else if (status === "ready") {
      clearTimeout(timeoutId);
      setStatus("too_soon");
    } else if (status === "now") {
      const rt = Date.now() - startTime;
      setReactionTime(rt);
      setStatus("result");

      const newLog = [...log, { timestamp: new Date().toISOString(), reactionTime: rt }];
      setLog(newLog);

      fetch("http://localhost:8000/log-reaction", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reaction_time_ms: rt }),
      });
    } else if (status === "too_soon" || status === "result") {
      setStatus("waiting");
      setReactionTime(null);
    }
  };

  const getMessage = () => {
    switch (status) {
      case "waiting":
        return "Click to start test";
      case "ready":
        return "Wait for green...";
      case "now":
        return "CLICK NOW!";
      case "result":
        return `Your reaction time: ${reactionTime} ms. Click to retry.`;
      case "too_soon":
        return "Too soon! Click to try again.";
      default:
        return "";
    }
  };

  return (
    <Card className="p-4 text-center">
      <CardContent>
        <h2 className="text-xl font-bold mb-4">🎯 Reaction Time Game</h2>
        <div
          className={`h-32 flex items-center justify-center rounded-xl cursor-pointer ${
            status === "now" ? "bg-green-400" : status === "ready" ? "bg-yellow-300" : "bg-gray-200"
          }`}
          onClick={handleClick}
        >
          <span className="text-lg font-medium">{getMessage()}</span>
        </div>
        {log.length > 0 && (
          <div className="mt-4 text-left">
            <h3 className="font-semibold">Recent Results:</h3>
            <ul className="text-sm list-disc pl-4">
              {log.slice(-5).map((entry, idx) => (
                <li key={idx}>{entry.reactionTime} ms @ {new Date(entry.timestamp).toLocaleTimeString()}</li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ReactionGame;

