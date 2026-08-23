/** Format an ISO timestamp for display; never renders "Invalid Date".
 *  Returns '—' for null/undefined/unparseable values (e.g. optimistic
 *  project objects that don't carry created_at yet). */
export function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleDateString();
}

/** Same, but with time-of-day (project detail pages). */
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleString();
}
