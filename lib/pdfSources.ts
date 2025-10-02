export const NFPA13_URL =
  "https://edufire.ir/storage/Library/ETFA-ABI/NFPA/NFPA%2013-2022.pdf";
export const PCI13R_URL =
  "https://www.pci.org/pci_docs/Design_Resources/Building_Engineering_Resources/NFPA_13R.pdf";

export function sourceUrl(docName: string, page: number) {
  if (/13-2022/i.test(docName)) return `${NFPA13_URL}#page=${page}`;
  if (/13R/i.test(docName)) return `${PCI13R_URL}#page=${page}`;
  return "#";
}

export type Hit = {
  doc: string;
  page: number;
  text: string;
  score: number;
};

export type AskResponse = {
  answer: string;
  hits: Hit[];
};
