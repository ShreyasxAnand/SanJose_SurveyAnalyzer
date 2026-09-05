import { useEffect, useRef, useState } from "react";
import { registerPasscodeAsker } from "./api";
import "./passcode.css";

/* The masked passcode dialog, mounted once at the app root.

   It replaces a window.prompt. The prompt worked — it is synchronous, so it
   could pause an in-flight request until someone typed — but a browser prompt
   cannot mask its input, which meant the passcode was displayed in the clear
   every time an action tripped the gate. This registers itself with api.ts as
   the way to ask, and resolves the promise the 401 handler is waiting on.

   Deliberately NOT built on Catalog's Modal: that one lives inside `.cat`,
   whose `transform: translateX(-50%)` would become the containing block for a
   `position: fixed` backdrop and break `inset: 0`. This renders at the app
   root, so it carries its own tokens — the same thing ask.css and
   pipeline.css already do. */

type Ask = {
  reason: string;
  resolve: (passcode: string | null) => void;
};

export default function PasscodeGate() {
  const [ask, setAsk] = useState<Ask | null>(null);
  /* Only WHETHER something has been typed, never what. The field is
     uncontrolled and read from the ref on submit, so React never writes the
     passcode into the `value` attribute — which means a DOM serialization
     (an error reporter, a saved page, an accessibility dump) cannot pick it
     up. Masking hides it from the screen; this keeps it out of the markup. */
  const [hasValue, setHasValue] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);
  // what had focus before the dialog stole it, so it can be handed back
  const returnFocusTo = useRef<Element | null>(null);

  useEffect(() => {
    registerPasscodeAsker(
      ({ reason }) =>
        new Promise<string | null>((resolve) => {
          returnFocusTo.current = document.activeElement;
          setAsk({ reason, resolve });
        }),
    );
    return () => registerPasscodeAsker(null);
  }, []);

  useEffect(() => {
    if (ask) inputRef.current?.focus();
  }, [ask]);

  if (!ask) return null;

  const finish = (passcode: string | null) => {
    /* Resolve before clearing: the awaiting request should be unblocked even
       if a re-render throws. Cancelling resolves null rather than hanging —
       the caller then returns the original 401 and the error banner explains
       it, which is the behaviour a dismissed prompt used to have. */
    ask.resolve(passcode);
    setAsk(null);
    if (inputRef.current) inputRef.current.value = "";
    setHasValue(false);
    if (returnFocusTo.current instanceof HTMLElement) {
      returnFocusTo.current.focus();
    }
  };

  return (
    <div
      className="pg-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) finish(null);
      }}
    >
      <div
        className="pg-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="pg-title"
        onKeyDown={(e) => {
          if (e.key === "Escape") finish(null);
        }}
      >
        <h3 id="pg-title">Admin passcode</h3>
        <p className="pg-reason">{ask.reason}</p>
        <label className="pg-field">
          <span>Passcode</span>
          <input
            ref={inputRef}
            type="password"
            defaultValue=""
            autoComplete="current-password"
            onChange={(e) => setHasValue(e.target.value.trim() !== "")}
            onKeyDown={(e) => {
              const typed = inputRef.current?.value.trim();
              if (e.key === "Enter" && typed) finish(typed);
            }}
          />
        </label>
        <p className="pg-hint">
          Kept for this browser tab only, and sent with actions that change
          data. Manage it on the Settings screen.
        </p>
        <div className="pg-actions">
          <button className="pg-btn" onClick={() => finish(null)}>
            Cancel
          </button>
          <button
            className="pg-btn pg-btn-primary"
            disabled={!hasValue}
            onClick={() => finish(inputRef.current?.value.trim() || null)}
          >
            Unlock
          </button>
        </div>
      </div>
    </div>
  );
}
