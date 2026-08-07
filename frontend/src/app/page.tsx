"use client";

import { useState } from "react";

const RACECOURSES = [
  "札幌", "函館", "福島", "新潟", "東京",
  "中山", "中京", "京都", "阪神", "小倉",
];

const MARKS: Record<string, string> = {
  "◎": "text-red-600 font-black text-2xl",
  "○": "text-blue-600 font-black text-2xl",
  "▲": "text-green-600 font-black text-2xl",
  "△": "text-purple-600 font-bold text-xl",
  "×": "text-gray-500 font-bold text-xl",
};

interface Horse {
  horse_num: number;
  gate_num: number;
  horse_name: string;
  jockey_name: string;
  win_prob: number;
  rank: number;
  mark: string;
}

interface PredictResult {
  race_info: {
    date: string;
    racecourse: string;
    race_num: number;
    distance: number;
    surface: string;
    track_condition: string;
    weather: string;
  };
  horses: Horse[];
  recommendation: Record<string, string>;
  predicted_at: string;
}

export default function Home() {
  const today = new Date().toISOString().split("T")[0];
  const [date, setDate] = useState(today);
  const [racecourse, setRacecourse] = useState("東京");
  const [raceNum, setRaceNum] = useState(11);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState("");

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  async function handlePredict() {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          date,
          racecourse,
          race_num: raceNum,
          kai: 1,
          day: 1,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "予想の取得に失敗しました");
      }
      const data = await res.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message || "エラーが発生しました");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gray-50 pb-20">
      {/* ヘッダー */}
      <div className="bg-green-700 text-white p-4 text-center">
        <h1 className="text-2xl font-bold">🏇 競馬AI予想</h1>
        <p className="text-sm opacity-80 mt-1">機械学習による本命・穴馬予想</p>
      </div>

      {/* 入力フォーム */}
      <div className="max-w-md mx-auto mt-6 px-4">
        <div className="bg-white rounded-2xl shadow p-5 space-y-4">
          {/* 日付 */}
          <div>
            <label className="block text-sm font-semibold text-gray-600 mb-1">開催日</label>
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-lg focus:outline-none focus:ring-2 focus:ring-green-500"
            />
          </div>

          {/* 競馬場 */}
          <div>
            <label className="block text-sm font-semibold text-gray-600 mb-1">競馬場</label>
            <div className="grid grid-cols-5 gap-2">
              {RACECOURSES.map((c) => (
                <button
                  key={c}
                  onClick={() => setRacecourse(c)}
                  className={`py-2 rounded-lg text-sm font-bold border-2 transition-all ${
                    racecourse === c
                      ? "bg-green-600 text-white border-green-600"
                      : "bg-white text-gray-700 border-gray-300"
                  }`}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>

          {/* レース番号 */}
          <div>
            <label className="block text-sm font-semibold text-gray-600 mb-1">レース番号</label>
            <div className="grid grid-cols-6 gap-2">
              {Array.from({ length: 12 }, (_, i) => i + 1).map((n) => (
                <button
                  key={n}
                  onClick={() => setRaceNum(n)}
                  className={`py-2 rounded-lg text-sm font-bold border-2 transition-all ${
                    raceNum === n
                      ? "bg-green-600 text-white border-green-600"
                      : "bg-white text-gray-700 border-gray-300"
                  }`}
                >
                  {n}R
                </button>
              ))}
            </div>
          </div>

          {/* 予想ボタン */}
          <button
            onClick={handlePredict}
            disabled={loading}
            className="w-full bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white font-black text-xl py-4 rounded-xl transition-colors"
          >
            {loading ? "予想中..." : "🎯 予想する"}
          </button>
        </div>

        {/* エラー */}
        {error && (
          <div className="mt-4 bg-red-50 border border-red-300 text-red-700 rounded-xl p-4 text-sm">
            {error}
          </div>
        )}

        {/* 予想結果 */}
        {result && (
          <div className="mt-6 space-y-4">
            {/* レース情報 */}
            <div className="bg-white rounded-2xl shadow p-4">
              <h2 className="font-bold text-lg text-gray-800 mb-2">
                {result.race_info.racecourse} {result.race_info.race_num}R
              </h2>
              <div className="flex gap-3 text-sm text-gray-500 flex-wrap">
                <span>{result.race_info.surface}{result.race_info.distance}m</span>
                <span>馬場: {result.race_info.track_condition}</span>
                <span>天気: {result.race_info.weather}</span>
              </div>
            </div>

            {/* 買い目 */}
            <div className="bg-yellow-50 border border-yellow-300 rounded-2xl shadow p-4">
              <h3 className="font-black text-gray-800 mb-3">💰 推奨買い目</h3>
              <div className="space-y-2">
                {Object.entries(result.recommendation).map(([type, bet]) => (
                  <div key={type} className="flex justify-between items-center">
                    <span className="text-sm font-bold text-gray-600">{type}</span>
                    <span className="font-black text-lg text-red-700">{bet}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* 全馬予想 */}
            <div className="bg-white rounded-2xl shadow p-4">
              <h3 className="font-bold text-gray-800 mb-3">📊 全馬予想スコア</h3>
              <div className="space-y-3">
                {result.horses.map((h) => (
                  <div key={h.horse_num} className="flex items-center gap-3">
                    <span className={`w-8 text-center ${MARKS[h.mark] || "text-gray-400"}`}>
                      {h.mark}
                    </span>
                    <span className="text-sm text-gray-400 w-8 text-center">
                      {h.horse_num}番
                    </span>
                    <span className="flex-1 font-bold text-gray-800">{h.horse_name}</span>
                    <span className="text-xs text-gray-400">{h.jockey_name}</span>
                    <div className="w-24">
                      <div className="flex items-center gap-1">
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-green-500 h-2 rounded-full"
                            style={{ width: `${Math.min(h.win_prob * 500, 100)}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-600 w-10 text-right">
                          {(h.win_prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <p className="text-center text-xs text-gray-400">
              予想生成: {new Date(result.predicted_at).toLocaleString("ja-JP")}
            </p>
          </div>
        )}
      </div>
    </main>
  );
}
