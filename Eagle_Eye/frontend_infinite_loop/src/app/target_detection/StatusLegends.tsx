import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function StatusLegends() {
  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader>
        <CardTitle>Status Legend</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex items-center space-x-3">
            <div className="h-4 w-4 rounded-full bg-red-500"></div>
            <span className="text-sm">Less than 6 hours</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="h-4 w-4 rounded-full bg-orange-500"></div>
            <span className="text-sm">Less than 1 day</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="h-4 w-4 rounded-full bg-yellow-500"></div>
            <span className="text-sm">Less than 1 week</span>
          </div>
          <div className="flex items-center space-x-3">
            <div className="h-4 w-4 rounded-full bg-gray-400"></div>
            <span className="text-sm">More than 1 week</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
