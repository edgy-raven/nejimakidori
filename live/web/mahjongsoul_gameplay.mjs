import { riichiOnlyDecision } from "./riichi_debug.mjs";
import { mkdir, rename, writeFile } from "node:fs/promises";
import { soulTile, discardTarget, offeredButtons } from "./mahjongsoul_targeting.mjs";
import { livePredictionBody } from "./live_prediction.mjs";

const callTypes = ["chi", "pon", "daiminkan"];
const callTypeSet = new Set(callTypes);
const kanTypeSet = new Set(["ankan", "kakan"]);

// Preserve the server's menu order, including nonpreferred red-free calls.
export function callComposition(action, operation) {
  if (!["chi", "pon"].includes(action.type)) return null;
  const type = action.type === "chi" ? 2 : 3;
  const combinations = operation.operation_list
    .filter(value => value.type === type)
    .flatMap(value => value.combination);
  const consumed = action.consumed.map(soulTile).sort().join("|");
  const index = combinations.findIndex(value =>
    value.split("|").sort().join("|") === consumed);
  if (index < 0 || combinations.length > 6) {
    throw new Error(`${action.type} composition was not offered or recognized`);
  }
  return combinations.length === 1 ? null : {
    x: 627 + Math.round((index - (combinations.length - 1) / 2) * 133),
    y: 510,
  };
}

// Shared hand inference, visual verification and tile input. Room workflows
// and authorization belong to the controller providing ctx.
export function createGameplay(ctx) {
  let expectedAction = null;
  let expectedActionAt = null;
  let expectedButton = null;
  let buttonReferenceWrite = Promise.resolve();
  function clickTarget(action) {
    if (action.type === "dahai") {
      return { control: "discard" };
    }
    const buttons = {
      none: { control: "skip" },
      chi: { control: "chii" },
      pon: { control: "pon" },
      reach: { control: "reach" },
      ankan: { control: "kan" },
      kakan: { control: "kan" },
      daiminkan: { control: "kan" },
      hora: { control: "hora" },
      ryukyoku: { control: "ryukyoku" },
    };
    if (!buttons[action.type]) {
      throw new Error(`automatic ${action.type} clicks are not calibrated`);
    }
    return buttons[action.type];
  }

  function expectedResult(action, body) {
    if (action.type === "dahai") {
      return action;
    }
    if (action.type === "reach") {
      return { ...body.riichi_discard, riichi: true };
    }
    if (callTypeSet.has(action.type) || kanTypeSet.has(action.type) ||
        ["hora", "ryukyoku"].includes(action.type)) {
      return action;
    }
    return null;
  }

  function verifyExpectedAction(action) {
    if (!expectedAction) {
      return;
    }
    if (callTypeSet.has(expectedAction.type) && (
      (action.name === "ActionChiPengGang" &&
        Number(action.data.seat) !== Number(expectedAction.actor) &&
        action.data.tiles.some((tile, index) =>
          Number(action.data.froms[index]) !== Number(action.data.seat) &&
          String(tile) === soulTile(expectedAction.pai))) ||
      (action.name === "ActionHule" && action.data.hules.some(winner =>
        Number(winner.seat) !== Number(expectedAction.actor))))) {
      ctx.output("call_superseded", {expected: expectedAction, actual: action});
      expectedAction = null;
      expectedActionAt = null;
      expectedButton = null;
      return;
    }
    let actual;
    let matches;
    if (
      action.name === "ActionDealTile" &&
      ["chi", "pon", "daiminkan"].includes(expectedAction.type)
    ) {
      actual = { type: "draw_before_call", seat: Number(action.data.seat) };
      matches = false;
    } else if (
      action.name === "ActionDiscardTile" &&
      Number(action.data.seat) === Number(expectedAction.actor)
    ) {
      actual = {
        tile: String(action.data.tile),
        riichi: Boolean(action.data.is_liqi || action.data.is_wliqi),
      };
      matches = expectedAction.type === "dahai" &&
        actual.tile === soulTile(expectedAction.pai) &&
        actual.riichi === Boolean(expectedAction.riichi);
    } else if (
      action.name === "ActionChiPengGang" &&
      Number(action.data.seat) === Number(expectedAction.actor)
    ) {
      actual = {
        type: callTypes[Number(action.data.type)],
        tiles: action.data.tiles.map(String).sort(),
      };
      matches = callTypeSet.has(expectedAction.type) &&
        actual.type === expectedAction.type &&
        JSON.stringify(actual.tiles) === JSON.stringify([
          ...expectedAction.consumed.map(soulTile),
          soulTile(expectedAction.pai),
        ].sort());
    } else if (
      action.name === "ActionAnGangAddGang" &&
      Number(action.data.seat) === Number(expectedAction.actor)
    ) {
      actual = {
        type: Number(action.data.type) === 3 ? "ankan" : "kakan",
        tile: String(action.data.tiles),
      };
      matches = kanTypeSet.has(expectedAction.type) &&
        actual.type === expectedAction.type &&
        actual.tile === soulTile(expectedAction.pai);
    } else if (action.name === "ActionLiuJu") {
      actual = { type: "ryukyoku" };
      matches = expectedAction.type === "ryukyoku";
    } else if (action.name === "ActionHule") {
      actual = { type: "hora" };
      matches = expectedAction.type === "hora" && action.data.hules.some(
        (winner) => Number(winner.seat) === Number(expectedAction.actor),
      );
    } else {
      return;
    }
    const verification = {expected: expectedAction, actual};
    ctx.output(matches ? "action_verified" : "action_mismatch", verification);
    if (matches && expectedButton) {
      const button = expectedButton;
      buttonReferenceWrite = buttonReferenceWrite.then(async () => {
        await mkdir(ctx.buttonCapturePath, { recursive: true });
        const path = `${ctx.buttonCapturePath}/${button.control}`;
        await writeFile(`${path}.json.tmp`, JSON.stringify({
          box: button.box, image: button.image.toString("base64"),
          verifiedBy: actual,
        }));
        await rename(`${path}.json.tmp`, `${path}.json`);
        ctx.output("button_reference_saved", { control: button.control });
      }).catch((error) => ctx.output("button_reference_error", {
        message: error.message,
      }));
    }
    expectedButton = null;
    expectedAction = null;
    if (!matches) ctx.handleActionMismatch(verification);
  }

  function thinkDelay(position, body) {
    const difficulty = Number(body.difficulty || 0);
    const transitionDelay = position.action === "ActionNewRound"
      ? 3200
      : (position.action === "ActionChiPengGang" ? 800 : 0);
    const delay = transitionDelay + (ctx.casual ? 0 : 50 + Math.floor(
      250 * difficulty ** 2 + Math.random() * 35,
    ));
    const timer = (
      Number(position.operation?.time_fixed || 0) +
      Number(position.operation?.time_add || 0)
    );
    const ceiling = timer
      ? Math.max(0, timer - (Date.now() - position.receivedAt) - 900)
      : delay;
    return Math.min(delay, ceiling);
  }

  function isActive(version) {
    return !ctx.automationBlock && ctx.canPlay() && version === ctx.positionVersion && ctx.autoplay &&
      ctx.liveState.status === "playing";
  }

  function controlReady(target, position, control) {
    if (target.control === "discard") {
      return control.hand.decision &&
        control.hand.meld_count === position.hand.meldCount;
    }
    return target.control in control.actions;
  }

  async function captureActionControl(target, position, version) {
    await ctx.send("Input.dispatchMouseEvent", {
      type: "mouseMoved",
      x: 640,
      y: 350,
    });
    await ctx.wait(120);
    let control;
    let ready = false;
    for (let attempt = 0; attempt < 4 && !ready; attempt += 1) {
      control = await ctx.captureControl("control", position);
      if (!isActive(version)) {
        return null;
      }
      ready = controlReady(target, position, control);
      if (!ready) {
        await ctx.wait(180);
      }
    }
    return { control, ready };
  }

  function screenSummary(control) {
    return {
      screen: control.screen,
      hand: {
        meldCount: control.hand.meld_count,
        decision: control.hand.decision,
      },
      actions: control.actions,
    };
  }

  function selectionTarget(action, body, position, control, target) {
    if (target.control === "reach") {
      return discardTarget(body.riichi_discard, position, control);
    }
    const kanChoices = new Set(
      (body.possible_actions || [])
        .filter((value) => kanTypeSet.has(value.type))
        .map((value) => `${value.type}:${value.pai}`),
    ).size;
    return target.control === "kan" && kanChoices > 1
      ? discardTarget({ type: "dahai", pai: action.pai }, position, control)
      : null;
  }

  async function verifiedDiscardTarget(action, position, control, version) {
    let previous = null;
    let verificationError;
    for (let attempt = 0; attempt < 8; attempt += 1) {
      try {
        const target = discardTarget(action, position, control);
        if (previous && target.x === previous.x && target.y === previous.y) {
          return target;
        }
        previous = target;
        verificationError = null;
      } catch (error) {
        if (!error.message.startsWith("screen slot for ")) throw error;
        previous = null;
        verificationError = error;
        ctx.output("discard_verification_retry", {
          attempt: attempt + 1,
          reason: error.message,
          hand: position.hand,
          screenTiles: control.hand.tiles.map(
            ({ label, trusted, drawn, confidence, glare }) =>
              ({ label, trusted, drawn, confidence, glare }),
          ),
        });
      }
      await ctx.wait(350);
      control = await ctx.captureControl("control", position);
      if (!isActive(version)) return null;
    }
    throw verificationError || new Error("screen slot for discard never settled");
  }

  async function resolveTarget(action, position, control, target, version) {
    if (target.control === "discard") {
      return verifiedDiscardTarget(action, position, control, version);
    }
    const [x, y] = control.actions[target.control].center;
    return { ...target, x, y };
  }

  function jitter(target) {
    return {
      x: target.x + Math.floor(Math.random() * 13) - 6,
      y: target.y + Math.floor(Math.random() * 11) - 5,
    };
  }

  async function selectCallComposition(composition, version) {
    if (!composition || !isActive(version)) return 0;
    await ctx.click(composition.x, composition.y);
    return 1;
  }

  async function clickSelectionTwice(target, version) {
    if (!isActive(version)) {
      return 0;
    }
    const point = jitter(target);
    await ctx.click(point.x, point.y);
    await ctx.wait(250);
    if (!isActive(version)) {
      return 1;
    }
    await ctx.click(point.x, point.y);
    return 2;
  }

  async function completeAction(action, body, target, selection, version, receivedAt, operation) {
    const composition = callComposition(action, operation);
    const point = jitter(target);
    expectedAction = expectedResult(action, body);
    expectedActionAt = Date.now();
    const started = performance.now();
    await ctx.click(point.x, point.y);
    ctx.output("click_timing", {
      action: action.type, dispatch_ms: performance.now() - started,
      position_to_click_ms: Date.now() - receivedAt,
    });
    await ctx.wait(350);
    let attempts = 1 + await selectCallComposition(composition, version);
    if (target.control === "kan" && selection) {
      attempts += await clickSelectionTwice(selection, version);
    }
    if (target.control === "reach") {
      attempts += await clickSelectionTwice(selection, version);
    }
    if (target.control === "discard" && isActive(version)) {
      await ctx.click(point.x, point.y);
      attempts += 1;
    }
    return attempts;
  }

  async function act(position, body, version) {
    if (!isActive(version)) {
      return;
    }
    const action = body.action;
    if (!action) {
      return;
    }
    position = {
      ...position,
      // Model phases can omit still-visible alternatives, such as kan at riichi.
      buttons: offeredButtons(position),
    };
    if (position.hand.riichi && action.type === "dahai") {
      expectedAction = expectedResult(action, body);
      expectedActionAt = Date.now();
      ctx.output("automatic_action_expected", { action });
      return;
    }
    try {
      const delay = thinkDelay(position, body);
      ctx.output("click_wait", {
        delay,
        difficulty: body.difficulty || 0,
        action,
      });
      await ctx.wait(delay);
      if (!isActive(version)) {
        return;
      }
      let target = clickTarget(action);
      const captured = await captureActionControl(target, position, version);
      if (!captured) {
        return;
      }
      const { control, ready } = captured;
      if (!control.control_ready || !ready) {
        await mkdir(`${ctx.buttonCapturePath}/unresolved`, { recursive: true });
        await writeFile(
          `${ctx.buttonCapturePath}/unresolved/${target.control}.json`,
          JSON.stringify({
            image: control.image.toString("base64"),
            position, action, detection: screenSummary(control),
          }),
        );
        ctx.pauseAutomation("The game controls could not be verified before a move.");
        ctx.output("click_refused", {
          reason: "screen verification failed",
          action,
          control: screenSummary(control),
        });
        return;
      }
      const tileSelectionTarget = selectionTarget(
        action, body, position, control, target,
      );
      if (!isActive(version)) {
        return;
      }
      target = await resolveTarget(
        action, position, control, target, version,
      );
      if (!target) {
        return;
      }
      expectedButton = target.control === "discard" ? null : {
        control: target.control,
        box: control.actions[target.control].box,
        image: control.image,
      };
      if (["chi", "pon"].includes(action.type)) {
        await mkdir(`${ctx.buttonCapturePath}/attempts`, { recursive: true });
        await writeFile(
          `${ctx.buttonCapturePath}/attempts/${target.control}.json`,
          JSON.stringify({
            image: control.image.toString("base64"),
            position, action, target, detection: screenSummary(control),
          }),
        );
        if (!isActive(version)) return;
      }
      const attempts = await completeAction(
        action, body, target, tileSelectionTarget, version, position.receivedAt, position.operation,
      );
      ctx.output("clicked", { action, delay, attempts, target });
    } catch (error) {
      if (!isActive(version)) {
        return;
      }
      ctx.pauseAutomation(error.message);
      ctx.output("click_refused", { reason: error.message, action });
    }
  }

  async function prediction(position, version) {
    if (
      !ctx.canPlay() || !position.phase || position.selfSeat === null ||
      version !== ctx.positionVersion
    ) {
      return;
    }
    const started = performance.now();
    const response = await fetch(`${ctx.modelUrl}/predict-live`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        round: position.round,
        actor: position.selfSeat,
      }),
    });
    const body = await response.json();
    ctx.output("prediction_timing", {
      step: position.step, phase: position.phase,
      queue_ms: Date.now() - position.receivedAt - (performance.now() - started),
      request_ms: performance.now() - started,
      model_ms: body.inference?.latency_ms,
    });
    if (version !== ctx.positionVersion) return;
    let result = response.ok
      ? livePredictionBody(body)
      : { error: body.error?.message || "Model request failed." };
    if (response.ok && ctx.riichiOnly) result = riichiOnlyDecision(position, result);
    ctx.output("prediction", {
      status: response.status,
      body: result,
    });
    ctx.publishLiveState({
      status: "playing",
      prediction: { status: response.status, body: result },
    });
    if (response.ok) {
      await act(position, result, version);
    } else {
      ctx.pauseAutomation(result.error);
    }
  }
  return {prediction, verifyExpectedAction,
    get expectedAction() {return expectedAction;},
    set expectedAction(value) {expectedAction = value;},
    get expectedActionAt() {return expectedActionAt;},
    set expectedButton(value) {expectedButton = value;},
    get buttonReferenceWrite() {return buttonReferenceWrite;},
  };
}
