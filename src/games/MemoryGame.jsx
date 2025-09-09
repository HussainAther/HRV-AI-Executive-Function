import React, { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const ALL_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ".split("");

export default function MemoryGame({ executiveFunctionScore = 0.5 }) {
  const [sequence, setSequence] = useState([]);
  const [currentLetter, setCurrentLetter] = useState("");
  const [userInput, setUserInput] = useState("");
  const [score, setScore] = useState(0);
  const [index, setIndex] = useState(0);
  const [nBack, setNBack] = useState(2); // adaptive

  // 🧠 Adjust game difficulty based on EF score
  useEffect(() => {
    const newNBack = Math.max(1, Math.round(2 + executiveFunctionScore * 3)); // Range: 2-5
    setNBack(newNBack);
  }, [executiveFunctionScore]);

  const generateSequence = (length = 20) => {
    const newSeq = Array.from({ length }, () => {
      return ALL_LETTERS[Math.floor(Math.random() * ALL_LETTERS.length)];
    });
    setSequence(newSeq);
    setIndex(0);
    setScore(0);
  };

  const handleInput = (match) => {
    if (index < nBack) return;
    const isMatch = sequence[index] === sequence[index - nBack];
    if ((match && isMatch) || (!match && !isMatch)) {
      setScore((s) => s + 1);
    }
    setIndex((i) => i + 1);
  };

  useEffect(() => {
    if (sequence.length > 0 && index < sequence.length) {
      setCurrentLetter(sequence[index]);
    }
  }, [index, sequence]);

  return (
    <Card>
      <CardContent className="p-6 space-y-4 text-center">
        <h2 className="text-xl font-bold">🧠 Adaptive N-Back Game</h2>
        <p className="text-sm text-muted-foreground">
          Current N-Back Level: {nBack}
        </p>
        {sequence.length === 0 ? (
          <Button onClick={() => generateSequence()}>Start Game</Button>
        ) : index >= sequence.length ? (
          <>
            <p>✅ Game Over</p>
            <p>Final Score: {score} / {sequence.length - nBack}</p>
            <Button onClick={() => generateSequence()}>Play Again</Button>
          </>
        ) : (
          <>
            <div className="text-4xl font-bold">{currentLetter}</div>
            <div className="flex justify-center gap-4 mt-4">
              <Button onClick={() => handleInput(true)}>Match</Button>
              <Button onClick={() => handleInput(false)}>No Match</Button>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

