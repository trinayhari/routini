import TestBackend from "@/components/TestBackend";
import { MainLayout } from "@/components/layout/main-layout";

export default function TestPage() {
  return (
    <MainLayout>
      <TestBackend />
    </MainLayout>
  );
}
