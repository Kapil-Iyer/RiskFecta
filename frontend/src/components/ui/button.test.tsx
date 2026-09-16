import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Button } from "./button";

describe("Button (shadcn foundation primitive)", () => {
  it("renders a native button by default and responds to clicks", async () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Click me</Button>);

    const button = screen.getByRole("button", { name: "Click me" });
    expect(button).toBeInTheDocument();
    button.click();
    expect(onClick).toHaveBeenCalledOnce();
  });

  it("renders as its child element via asChild (Radix Slot) instead of a nested button", () => {
    render(
      <Button asChild>
        <a href="/methodology">Go</a>
      </Button>,
    );

    const link = screen.getByRole("link", { name: "Go" });
    expect(link).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});
