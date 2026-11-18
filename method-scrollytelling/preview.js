(async function () {
  "use strict";
  const button = document.querySelector(".preview-open");
  try {
    const response = await fetch("method.html");
    if (!response.ok) throw new Error("method.html HTTP " + response.status);
    const fragment = new DOMParser().parseFromString(await response.text(), "text/html").querySelector("#tab-method");
    if (!fragment) throw new Error("Missing #tab-method");
    document.querySelector("#preview-mount").append(document.importNode(fragment, true));
    const root = document.querySelector("#tab-method");
    const method = initMethod(root);
    button.disabled = false;
    button.addEventListener("click", () => method.open(button));
    // Development harness only. It never enters the final HTML fragment.
    if (new URLSearchParams(location.search).get("qa") === "1") {
      const errors = [];
      window.addEventListener("error", (event) => errors.push(event.message || "resource error: " + event.target?.src), true);
      window.addEventListener("unhandledrejection", (event) => errors.push(String(event.reason)));
      Object.defineProperty(root, "previewQA", { value: { method, errors }, configurable: true });
      // Optional development deep link: ?qa=1&step=1-2&progress=0.5.
      // It uses the same QA controls, which change the real scroll position.
      const params = new URLSearchParams(location.search);
      if (Array.from(root.querySelectorAll('[data-step-id]')).some(step => step.dataset.stepId === params.get('step'))) {
        await method.open(button);
        const panel = root.querySelector(".method-qa");
        panel.querySelector("select").value = params.get("step");
        panel.querySelector('[data-qa-action="go"]').click();
        const p = params.get("progress") || "0.5";
        if (["0", "0.5", "1"].includes(p)) panel.querySelector('[data-qa-progress-target="' + p + '"]').click();
      }
    }
  } catch (error) {
    const notice = document.createElement("p");
    notice.className = "preview-error";
    notice.textContent = "미리보기를 불러오지 못했습니다. 이 폴더에서 HTTP 서버를 실행해주세요. " + error.message;
    document.querySelector(".preview-shell").append(notice);
    console.error("[method-preview]", error);
  }
}());
