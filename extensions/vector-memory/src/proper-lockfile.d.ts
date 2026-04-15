declare module "proper-lockfile" {
  function lock(path: string, opts?: unknown): Promise<() => Promise<void>>;
  export default { lock };
  export { lock };
}
