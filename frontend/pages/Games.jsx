import MemoryGame from "../games/MemoryGame";

export default function GamesPage() {
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Cognitive Games</h1>
      <MemoryGame />
    </div>
  );
}

