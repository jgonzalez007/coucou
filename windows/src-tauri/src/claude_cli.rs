// Chat through the Claude Code CLI — the same idea as codex_plan.rs: `claude`
// answers with its own sign-in, so Coucou stores no key, reads no credential
// and makes no network call of its own.
//
// `claude -p <prompt> --output-format json` answers with one JSON object on
// stdout. Only the parsing lives here as a pure function; the spawn wraps it.

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use serde_json::{json, Value};

use crate::chat::{self, Chat, ChatContext, ChatReply};
use crate::i18n::t;
use crate::platform;

/// The CLI starts slowly (it loads its own context first), so it gets longer
/// than the 90 s the Messages API gets in claude.rs.
const TIMEOUT: Duration = Duration::from_secs(120);
/// Everything we are willing to read from one answer.
const MAX_OUTPUT: usize = 4 * 1024 * 1024;

/// The provider id the settings and the island's picker use (providers.ts).
pub const ID: &str = "claude-cli";
/// What the CLI runs when the picker has no choice saved for it.
pub const DEFAULT_MODEL: &str = "claude-opus-5";

/// What one CLI answer carries: the text to show, and the session the next
/// turn resumes.
#[derive(Debug, PartialEq)]
pub struct Reply {
    pub text: String,
    pub session_id: Option<String>,
}

/// The session the CLI should resume: the one its own last answer left. A
/// provider switch rebuilds the history from the plain turns (chat.rs), which
/// carry no session — then there is none and the earlier turns go in the
/// prompt instead.
pub fn session_of(history: &[Value]) -> Option<String> {
    history
        .iter()
        .rev()
        .find_map(|message| message.get("session_id").and_then(Value::as_str))
        .map(str::to_string)
}

/// The prompt for one turn. With a session to resume the CLI already knows the
/// conversation, so only the question goes; without one the earlier turns go
/// in front of it, or the context would be lost on a provider switch.
pub fn prompt_with(history: &[Value], question: &str) -> String {
    if history.is_empty() || session_of(history).is_some() {
        return question.to_string();
    }
    let mut out = String::new();
    for message in history {
        let Some(content) = message.get("content").and_then(Value::as_str) else { continue };
        let who = match message.get("role").and_then(Value::as_str) {
            Some("assistant") => "Assistant",
            _ => "User",
        };
        out.push_str(who);
        out.push_str(": ");
        out.push_str(content);
        out.push('\n');
    }
    out.push('\n');
    out.push_str(question);
    out
}

/// The answer in the CLI's JSON, or the message the island shows instead.
pub fn answer(stdout: &[u8]) -> Result<Reply, String> {
    let answer: Value = serde_json::from_slice(stdout).map_err(|_| t("Claude Code gave no answer."))?;
    // Trimmed and non-empty, or there is nothing to show.
    let result = answer
        .get("result")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty());
    // The CLI reports its own failures in this same envelope, with its message
    // in `result`: a low balance, a model it cannot reach, a refused prompt.
    if answer.get("is_error").and_then(Value::as_bool) == Some(true) {
        return Err(match result {
            Some(why) => why.to_string(),
            None => t("Claude Code reported an error."),
        });
    }
    Ok(Reply {
        text: result.ok_or_else(|| t("No response text."))?.to_string(),
        session_id: answer.get("session_id").and_then(Value::as_str).map(str::to_string),
    })
}

/// Runs the CLI once and returns its stdout. Blocking: call it off the main
/// thread. Nothing of the user's is read and no key is passed — `claude`
/// answers with its own sign-in.
fn run(prompt: &str, model: &str, resume: Option<&str>) -> Result<Vec<u8>, String> {
    let exe = platform::claude_candidates()
        .into_iter()
        .next()
        .ok_or_else(|| t("Claude Code was not found. Install it to chat with it."))?;

    let mut cmd = Command::new(&exe);
    cmd.arg("-p").arg(prompt).args(["--output-format", "json"]).args(["--model", model]);
    if let Some(session) = resume {
        cmd.args(["--resume", session]);
    }
    // Somewhere with no project of its own: started in a folder with a
    // CLAUDE.md the CLI would load it into every single message.
    cmd.current_dir(std::env::temp_dir());
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
    // An npm install is a `.cmd` calling node: its own folder first on PATH,
    // so it finds its Node (as codex_plan.rs does).
    if let Some(dir) = exe.parent() {
        let mut dirs = vec![dir.to_path_buf()];
        if let Some(path) = std::env::var_os("PATH") {
            dirs.extend(std::env::split_paths(&path));
        }
        if let Ok(joined) = std::env::join_paths(dirs) {
            cmd.env("PATH", joined);
        }
    }
    platform::no_console(&mut cmd);

    let mut child = cmd.spawn().map_err(|_| t("Claude Code could not be started."))?;
    let answered = (|| {
        let stdout = child.stdout.take()?;
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut buffer = Vec::new();
            let _ = stdout.take(MAX_OUTPUT as u64).read_to_end(&mut buffer);
            let _ = tx.send(buffer);
        });
        rx.recv_timeout(TIMEOUT).ok()
    })();
    // A `.cmd` wrapper would leave the node process under it running: end the
    // whole tree, exactly as codex_plan.rs does.
    #[cfg(windows)]
    {
        let mut kill = Command::new("taskkill");
        kill.args(["/T", "/F", "/PID", &child.id().to_string()]);
        // It answered and exited on its own nearly every time: taskkill then
        // prints "process not found", which is not ours to show.
        kill.stdout(Stdio::null()).stderr(Stdio::null());
        platform::no_console(&mut kill);
        let _ = kill.status();
    }
    let _ = child.kill();
    let _ = child.wait();
    answered.ok_or_else(|| t("Claude Code took too long to answer."))
}

/// One chat turn through the CLI. The same signature as claude::send(), so
/// chat.rs reaches it the same way.
pub async fn send(
    chat: &Chat,
    model: &str,
    query: String,
    context: Option<ChatContext>,
) -> Result<ChatReply, String> {
    let turn = chat.begin(ID);
    let plain = chat::plain_question(turn.first, context.as_ref(), &query);
    let prompt = prompt_with(&turn.history, &plain);
    let resume = session_of(&turn.history);
    let model = model.to_string();

    let output = tauri::async_runtime::spawn_blocking(move || run(&prompt, &model, resume.as_deref()))
        .await
        .map_err(|_| t("Claude Code could not be started."))??;
    let reply = answer(&output)?;

    // The session rides in the answer, so the next turn resumes instead of
    // sending the whole conversation again.
    chat.commit(
        &turn,
        json!({ "role": "user", "content": plain }),
        json!({ "role": "assistant", "content": reply.text, "session_id": reply.session_id }),
        &plain,
        &reply.text,
    );
    Ok(ChatReply { text: reply.text })
}

/// The paths that only the real CLI can show: spawning it, reading what it
/// actually prints, and resuming a session. Ignored like speech::with_models —
/// they need Claude Code installed and signed in, and each one spends a
/// request. Run them with `cargo test --lib -- --ignored`.
#[cfg(test)]
mod with_cli {
    use super::*;

    #[test]
    #[ignore]
    fn it_is_found_spawned_and_its_answer_read() {
        let found = platform::claude_candidates();
        assert!(!found.is_empty(), "no claude CLI on this machine");

        let output = run("Reply with exactly: PROBE_OK", DEFAULT_MODEL, None).expect("the CLI ran");
        let reply = answer(&output).expect("its answer parsed");
        assert_eq!(reply.text, "PROBE_OK");
        assert!(reply.session_id.is_some(), "no session came back to resume");
    }

    #[test]
    #[ignore]
    fn a_resumed_session_still_knows_the_conversation() {
        let first = answer(&run("Remember the number 42. Reply with exactly: STORED", DEFAULT_MODEL, None).unwrap())
            .expect("the first answer");
        let session = first.session_id.expect("a session to resume");

        let output = run(
            "What number did I ask you to remember? Answer with the number only.",
            DEFAULT_MODEL,
            Some(&session),
        )
        .expect("the resumed run");
        let second = answer(&output).expect("the second answer");
        assert!(second.text.contains("42"), "the resumed session lost the conversation: {}", second.text);
    }

    /// The riskiest path in the change: chat.rs rebuilds the history from the
    /// plain turns when another provider answered last, which drops the session
    /// id — so prompt_with has to carry the transcript instead.
    #[test]
    #[ignore]
    fn a_provider_switch_keeps_the_context() {
        let chat = Chat::default();
        let earlier = chat.begin(chat::ANTHROPIC);
        chat.commit(
            &earlier,
            json!({ "role": "user", "content": "My favourite number is 17." }),
            json!({ "role": "assistant", "content": "Noted: 17." }),
            "My favourite number is 17.",
            "Noted: 17.",
        );

        let reply = tauri::async_runtime::block_on(send(
            &chat,
            DEFAULT_MODEL,
            "What is my favourite number? Answer with the number only.".into(),
            None,
        ))
        .expect("the CLI answered");
        assert!(reply.text.contains("17"), "the switch lost the context: {}", reply.text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn out(value: Value) -> Vec<u8> {
        serde_json::to_vec(&value).unwrap()
    }

    #[test]
    fn the_answer_and_its_session_come_from_the_cli_json() {
        let stdout = out(json!({
            "type": "result",
            "subtype": "success",
            "is_error": false,
            "result": "Paris is the capital of France.",
            "session_id": "dca780a3-ea71-42d0-a5b3-97f441d265b7",
            "num_turns": 1,
        }));
        assert_eq!(
            answer(&stdout).unwrap(),
            Reply {
                text: "Paris is the capital of France.".into(),
                session_id: Some("dca780a3-ea71-42d0-a5b3-97f441d265b7".into()),
            }
        );
        // An answer without a session still shows: only resuming is lost.
        let stdout = out(json!({ "is_error": false, "result": "Hi" }));
        assert_eq!(answer(&stdout).unwrap(), Reply { text: "Hi".into(), session_id: None });
        // Surrounding blank lines are not part of the answer.
        let stdout = out(json!({ "is_error": false, "result": " Hi \n" }));
        assert_eq!(answer(&stdout).unwrap().text, "Hi");
    }

    /// The history as chat.rs keeps it for this provider.
    fn turns(pairs: &[(&str, &str, Option<&str>)]) -> Vec<Value> {
        let mut out = Vec::new();
        for (question, answer, session) in pairs {
            out.push(json!({ "role": "user", "content": question }));
            match session {
                Some(id) => out.push(json!({ "role": "assistant", "content": answer, "session_id": id })),
                None => out.push(json!({ "role": "assistant", "content": answer })),
            }
        }
        out
    }

    /// Every message this module shows has a translation: a key that does not
    /// match extra.json silently shows in English, which no test would catch.
    #[test]
    fn its_messages_are_translated() {
        crate::i18n::set_for_test("fr");
        for english in [
            "Claude Code could not be started.",
            "Claude Code gave no answer.",
            "Claude Code reported an error.",
            "Claude Code took too long to answer.",
            "Claude Code was not found. Install it to chat with it.",
            "No response text.",
        ] {
            assert_ne!(t(english), english, "untranslated: {english}");
        }
        // Reached through the parsing, not just looked up.
        assert_eq!(answer(b"not json").unwrap_err(), "Claude Code n'a donné aucune réponse.");
        assert_eq!(
            answer(&out(json!({ "is_error": true }))).unwrap_err(),
            "Claude Code a signalé une erreur."
        );
        crate::i18n::set_for_test("en");
        assert_eq!(answer(b"not json").unwrap_err(), "Claude Code gave no answer.");
    }

    #[test]
    fn the_session_to_resume_comes_from_the_last_answer() {
        assert_eq!(session_of(&[]), None);
        let history = turns(&[("a", "b", Some("s1")), ("c", "d", Some("s2"))]);
        assert_eq!(session_of(&history).as_deref(), Some("s2"));
        // Rebuilt by a provider switch: plain turns carry no session.
        assert_eq!(session_of(&turns(&[("a", "b", None)])), None);
    }

    #[test]
    fn without_a_session_the_earlier_turns_go_in_the_prompt() {
        // A session to resume: the CLI has the conversation already.
        let history = turns(&[("a", "b", Some("s1"))]);
        assert_eq!(prompt_with(&history, "next"), "next");
        // No session and no history: just the question.
        assert_eq!(prompt_with(&[], "only"), "only");
        // No session but earlier turns: they go in front, so nothing is lost.
        let rebuilt = turns(&[("what is 2+2?", "4", None)]);
        assert_eq!(
            prompt_with(&rebuilt, "and times 3?"),
            "User: what is 2+2?\nAssistant: 4\n\nand times 3?"
        );
    }

    #[test]
    fn an_error_flag_or_unreadable_output_is_an_error() {
        // The CLI reports its own failures in the same envelope.
        let stdout = out(json!({ "is_error": true, "result": "Credit balance too low" }));
        assert_eq!(answer(&stdout).unwrap_err(), "Credit balance too low");
        // A flagged error with nothing to say still says something.
        assert!(!answer(&out(json!({ "is_error": true }))).unwrap_err().is_empty());
        // Not JSON at all: the CLI died before it answered.
        assert!(answer(b"claude: command failed").is_err());
        assert!(answer(b"").is_err());
        // JSON, but not an answer.
        assert!(answer(&out(json!({ "type": "result" }))).is_err());
        assert!(answer(&out(json!({ "is_error": false, "result": "  " }))).is_err());
    }
}
