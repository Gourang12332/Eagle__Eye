import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function App() {
  return (
    <div className="justify-center items-center h-screen w-screen flex flex-col gap-12">
      <Link href="/home/">
        <Button>Home Page</Button>
      </Link>
      <Link href="/target_detection/">
        <Button>Target Detection Mode</Button>
      </Link>
      <Link href="/target_detection/demo/">
        <Button>Demo cctv upload page</Button>
      </Link>
      <Link href="/restricted_mode/">
        <Button>Restricted Mode</Button>
      </Link>
      <Link href="/crowd_management/">
        <Button>Crowd Management Mode</Button>
      </Link>
    </div>
  );
}
