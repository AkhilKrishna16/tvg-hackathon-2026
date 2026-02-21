import { NextResponse } from "next/server";
import { sampleHeatmap } from "@/lib/data";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return NextResponse.json(sampleHeatmap(2_000));
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
