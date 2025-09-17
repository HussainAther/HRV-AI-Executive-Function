import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";

const patterns = [
  ["🟥", "🟦", "🟩"],
  ["🟦", "🟩", "🟨"],
  ["🟨", "🟪", "🟥"],
  ["🟩", "🟥", "🟪", "🟦"],
  ["🟥", "🟦", "🟨", "🟪", "🟩"],
];

export default function MemoryGame() {
  const [sequence, setSequence] = useState([]);
  const [index, setIndex] = useState(0);
  const [input, setInput] = useState([]);
  const [streak, setStreak] = useState(0);
  const [maxStreak, setMaxStreak] = useState(0);
  const [message, setMessage] = useState("");
  const [adaptive, setAdaptive] = useState(true);

  const logStreakServer = async (current, max) => {
    const entry = {
      streak: current,
      max_streak: max,
      timestamp: new Date().toISOString(),
      session_id: "session_" + Date.now(),
    };

    await fetch("http://localhost:8000/log-streak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entry),
    });
  };

  const startNewGame = () => {
    const newSeq = adaptive
      ? [...Array(Math.min(streak + 3, 8)).keys()].map(() =>
          ["🟥", "🟦", "🟩", "🟨", "🟪"][Math.floor(Math.random() * 5)]
        )
      : patterns[Math.floor(Math.random() * patterns.length)];

    setSequence(newSeq);
    setIndex(0);
    setInput([]);
    setMessage("Remember this pattern:");
    setTimeout(() => setMessage("Repeat it:"), 2000);
  };

  const handleInput = (color) => {
    const newInput = [...input, color];
    setInput(newInput);

    if (color !== sequence[newInput.length - 1]) {
      setMessage("❌ Incorrect! Game over.");
      logStreakServer(streak, maxStreak);
      setStreak(0);
      return;
    }

    if (newInput.length === sequence.length) {
      const newStreak = streak + 1;
      setStreak(newStreak);
      if (newStreak > maxStreak) setMaxStreak(newStreak);
      logStreakServer(newStreak, Math.max(newStreak, maxStreak));
      setMessage("✅ Correct! Next round...");
      setTimeout(startNewGame, 1000);
    }
  };

  useEffect(() => {
    startNewGame();
  }, []);

  return (
    <div className="p-4 max-w-xl mx-auto space-y-4">
      <h2 className="text-2xl font-bold">🧠 Memory Challenge</h2>

      <p className="text-md">Streak: {streak} | Max Streak: {maxStreak}</p>

      <div className="space-x-2 mb-2">
        <Button onClick={startNewGame}>🔄 New Game</Button>
        <Button onClick={() => setAdaptive((a) => !a)}>
          {adaptive ? "🧩 Adaptive: On" : "🧩 Adaptive: Off"}
        </Button>
      </div>

      <div className="border p-4 rounded-md bg-gray-100 min-h-[80px]">
        <p className="text-lg font-semibold">{message}</p>
        {message.startsWith("Remember") && (
          <div className="mt-2 flex space-x-2 text-2xl">
            {sequence.map((s, i) => (
              <span key={i}>{s}</span>
            ))}
          </div>
        )}
      </div>

      <div className="mt-4 flex space-x-2">
        {["🟥", "🟦", "🟩", "🟨", "🟪"].map((color) => (
          <Button
            key={color}
            onClick={() => handleInput(color)}
            className="text-xl"
          >
            {color}
          </Button>
        ))}
      </div>
    </div>
  );
}

